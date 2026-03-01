#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

// Windows socket headers
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const int    HOT_THRESHOLD = 220;
const double MIN_AREA = 500.0;
const double HAND_MIN_AREA = 6000.0;
const int    HTTP_PORT = 8082;

enum VisionMode {
  MODE_ORIGINAL,
  MODE_THERMAL,
  MODE_NIGHTVISION
};

// ---------------------------------------------------------------------------
// Shared frame buffer between main loop and HTTP server thread
// ---------------------------------------------------------------------------

struct SharedFrame {
  std::mutex          mtx;
  std::vector<uchar>  jpegBuf;   // latest JPEG-encoded frame
  bool                ready = false;
};

static SharedFrame g_frame;
static std::atomic<bool> g_running{ true };

// ---------------------------------------------------------------------------
// HTTP MJPEG server (runs in a separate thread)
// Serves a single persistent connection as multipart/x-mixed-replace.
// Open http://127.0.0.1:8080 in Chrome or Firefox.
// http://82.64.237.163:8082
// ---------------------------------------------------------------------------

void httpServerThread()
{
  WSADATA wsa{};
  WSAStartup(MAKEWORD(2, 2), &wsa);

  SOCKET listenSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listenSock == INVALID_SOCKET) {
    std::cerr << "[HTTP] socket() failed." << std::endl;
    return;
  }

  int opt = 1;
  setsockopt(listenSock, SOL_SOCKET, SO_REUSEADDR,
    reinterpret_cast<const char*>(&opt), sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(HTTP_PORT);
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
    std::cerr << "[HTTP] bind() failed on port " << HTTP_PORT << std::endl;
    closesocket(listenSock);
    return;
  }

  listen(listenSock, 5);
  std::cout << "[HTTP] MJPEG stream ready at http://82.64.237.163:" << HTTP_PORT << std::endl;

  while (g_running) {
    // Accept one client at a time (blocking with timeout via select)
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(listenSock, &fds);
    timeval tv{ 1, 0 };   // 1-second timeout so we can check g_running
    if (select(0, &fds, nullptr, nullptr, &tv) <= 0) continue;

    SOCKET client = accept(listenSock, nullptr, nullptr);
    if (client == INVALID_SOCKET) continue;

    // Drain the HTTP request (we don't need to parse it)
    char reqBuf[2048] = {};
    recv(client, reqBuf, sizeof(reqBuf) - 1, 0);

    // Send HTTP response header for MJPEG stream
    const char* httpHeader =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
      "Cache-Control: no-cache\r\n"
      "Connection: close\r\n"
      "\r\n";
    send(client, httpHeader, (int)strlen(httpHeader), 0);

    // Stream frames until client disconnects or app exits
    while (g_running) {
      std::vector<uchar> jpg;
      {
        std::lock_guard<std::mutex> lock(g_frame.mtx);
        if (!g_frame.ready) continue;
        jpg = g_frame.jpegBuf;
      }

      // Build MIME part header
      char partHeader[256];
      int partLen = snprintf(partHeader, sizeof(partHeader),
        "--frame\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: %d\r\n"
        "\r\n",
        (int)jpg.size());

      int r1 = send(client, partHeader, partLen, 0);
      int r2 = send(client, reinterpret_cast<const char*>(jpg.data()),
        (int)jpg.size(), 0);
      int r3 = send(client, "\r\n", 2, 0);

      if (r1 == SOCKET_ERROR || r2 == SOCKET_ERROR || r3 == SOCKET_ERROR)
        break;   // client disconnected

      std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 fps
    }

    closesocket(client);
  }

  closesocket(listenSock);
  WSACleanup();
}

// ---------------------------------------------------------------------------
// Gesture classification (unchanged)
// ---------------------------------------------------------------------------

std::string classifyGesture(const std::vector<cv::Point>& contour,
  const std::vector<cv::Vec4i>& defects,
  const cv::Rect& bbox)
{
  if (defects.empty()) return "FIST";

  int fingerCount = 0;

  for (const auto& defect : defects) {
    cv::Point start = contour[defect[0]];
    cv::Point end = contour[defect[1]];
    cv::Point farPt = contour[defect[2]];
    float     depth = defect[3] / 256.0f;

    if (depth < 20.0f) continue;

    double a = cv::norm(end - start);
    double b = cv::norm(farPt - start);
    double c = cv::norm(end - farPt);
    double denom = 2.0 * b * c;
    if (denom < 1e-6) continue;

    double angle = std::acos(
      std::clamp((b * b + c * c - a * a) / denom, -1.0, 1.0)
    ) * 180.0 / CV_PI;

    if (angle < 90.0) fingerCount++;
  }

  if (fingerCount == 0) return "FIST";
  if (fingerCount == 1) return "PEACE / POINT";
  if (fingerCount == 2) return "THREE FINGERS";
  if (fingerCount == 3) return "FOUR FINGERS";
  if (fingerCount >= 4) return "OPEN HAND";

  return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Normal mode: skin-based hand detection + gesture classification (unchanged)
// ---------------------------------------------------------------------------

void processModeNormal(const cv::Mat& frame,
  cv::Mat& output,
  cv::CascadeClassifier& faceCascade)
{
  cv::Mat hsv, skinMask, blurred, gray;

  cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
  cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);

  cv::inRange(hsv,
    cv::Scalar(0, 10, 60),
    cv::Scalar(10, 255, 255),
    skinMask);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, kernel);
  cv::GaussianBlur(skinMask, skinMask, cv::Size(3, 3), 0);

  std::vector<cv::Rect> faces;
  faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(60, 60));
  for (const auto& face : faces) {
    int pad = (int)(face.width * 0.15);
    int ex = std::max(0, face.x - pad);
    int ey = std::max(0, face.y - pad);
    int ew = std::min(skinMask.cols - ex, face.width + 2 * pad);
    int eh = std::min(skinMask.rows - ey, face.height + 2 * pad);
    cv::Rect expanded(ex, ey, ew, eh);
    skinMask(expanded) = 0;
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  for (const auto& face : faces) {
    cv::rectangle(output, face, cv::Scalar(255, 100, 0), 2);
    cv::putText(output, "FACE",
      cv::Point(face.x, face.y - 8),
      cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 100, 0), 1);
  }

  int handCount = 0;
  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area < HAND_MIN_AREA) continue;

    cv::Rect bboxCheck = cv::boundingRect(contour);
    float    aspectRatio = (float)bboxCheck.width / (float)bboxCheck.height;
    if (aspectRatio > 1.4f) continue;

    handCount++;
    cv::Rect bbox = bboxCheck;

    std::vector<cv::Point> hull_points;
    std::vector<int>       hull_indices;
    cv::convexHull(contour, hull_points);
    cv::convexHull(contour, hull_indices);

    std::vector<cv::Vec4i> defects;
    if (hull_indices.size() > 3)
      cv::convexityDefects(contour, hull_indices, defects);

    std::string gesture = classifyGesture(contour, defects, bbox);

    cv::rectangle(output, bbox, cv::Scalar(0, 255, 0), 2);
    cv::drawContours(output,
      std::vector<std::vector<cv::Point>>{hull_points},
      -1, cv::Scalar(255, 0, 255), 2);

    cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    cv::drawMarker(output, center, cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 20, 2);

    cv::putText(output, gesture,
      cv::Point(bbox.x, bbox.y - 12),
      cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);

    std::string areaStr = "AREA: " + std::to_string((int)area) + "px";
    cv::putText(output, areaStr,
      cv::Point(bbox.x, bbox.y + bbox.height + 18),
      cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);
  }

  std::string handStr = "HANDS DETECTED: " + std::to_string(handCount);
  cv::putText(output, handStr,
    cv::Point(10, 55),
    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

  cv::putText(output, "MODE: NORMAL | GESTURE DETECTION",
    cv::Point(10, 30),
    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static const std::string STREAM_1 = "http://82.64.237.163:8082/mjpg/video.mjpg";
static const std::string STREAM_2 = "http://82.64.237.163:8083/mjpg/video.mjpg";

int main()
{
  int activeStream = 1;

  // --- Input: MJPEG network stream via FFMPEG --------------------------------
  cv::VideoCapture cap(STREAM_1, cv::CAP_FFMPEG);
  if (!cap.isOpened()) {
    std::cerr << "[ERROR] Cannot open stream: " << STREAM_1 << std::endl;
    return -1;
  }
  std::cout << "[INFO] Input stream 1 opened: " << STREAM_1 << std::endl;

  // --- Face cascade ----------------------------------------------------------
  cv::CascadeClassifier faceCascade;
  if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
    std::cerr << "[ERROR] File not found: haarcascade_frontalface_default.xml" << std::endl;
    return -1;
  }

  // --- Start HTTP MJPEG server in background thread -------------------------
  std::thread serverThread(httpServerThread);

  // --- State -----------------------------------------------------------------
  cv::Mat    frame, output;
  VisionMode currentMode = MODE_ORIGINAL;

  std::cout << "[INFO] Optic system ready." << std::endl;
  std::cout << "       [O] Normal+Gesture  [T] Thermal  [N] NVG+Targeting  [1] Stream 1  [2] Stream 2  [ESC] Exit" << std::endl;

  // --- Main loop -------------------------------------------------------------
  while (true) {
    cap >> frame;
    if (frame.empty()) {
      std::cerr << "[WARN] Empty frame — stream ended or connection lost." << std::endl;
      break;
    }

    frame.copyTo(output);

    // ===================== MODE: THERMAL =====================
    if (currentMode == MODE_THERMAL) {
      cv::Mat gray, blur, thermal, thresh;

      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      cv::GaussianBlur(gray, blur, cv::Size(7, 7), 0);
      cv::applyColorMap(blur, thermal, cv::COLORMAP_INFERNO);
      cv::threshold(blur, thresh, HOT_THRESHOLD, 255, cv::THRESH_BINARY);

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      int targetCount = 0;
      for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < MIN_AREA) continue;

        targetCount++;
        cv::Rect bbox = cv::boundingRect(contour);
        cv::rectangle(thermal, bbox, cv::Scalar(0, 255, 0), 2);

        cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        cv::drawMarker(thermal, center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 15, 1);

        std::string label = "HOTSPOT [" + std::to_string((int)area) + "px]";
        cv::putText(thermal, label,
          cv::Point(bbox.x, bbox.y - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
      }

      std::string countStr = "TARGETS: " + std::to_string(targetCount);
      cv::putText(thermal, countStr,
        cv::Point(10, 55),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

      cv::putText(thermal, "MODE: THERMAL",
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

      output = thermal;
    }
    // ===================== MODE: NVG =====================
    else if (currentMode == MODE_NIGHTVISION) {
      cv::Mat gray, clahe_img;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
      clahe->apply(gray, clahe_img);

      std::vector<cv::Rect> faces;
      faceCascade.detectMultiScale(clahe_img, faces, 1.1, 4, 0, cv::Size(40, 40));

      cv::Mat zero_channel = cv::Mat::zeros(frame.size(), CV_8UC1);
      std::vector<cv::Mat> channels = { zero_channel, clahe_img, zero_channel };
      cv::merge(channels, output);

      cv::Mat noise(frame.size(), CV_8UC3);
      cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(20));
      cv::add(output, noise, output);

      for (const auto& face : faces) {
        cv::rectangle(output, face, cv::Scalar(0, 0, 255), 2);
        cv::Point center(face.x + face.width / 2, face.y + face.height / 2);
        cv::drawMarker(output, center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 1);
        cv::putText(output, "TGT ACQUIRED",
          cv::Point(face.x, face.y - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
      }

      cv::putText(output, "MODE: NVG + TARGETING",
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    // ===================== MODE: NORMAL =====================
    else {
      processModeNormal(frame, output, faceCascade);
    }

    // --- Local preview window ----------------------------------------------
    std::string streamLabel = "SRC: STREAM " + std::to_string(activeStream);
    cv::putText(output, streamLabel,
      cv::Point(output.cols - 160, 30),
      cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(100, 200, 255), 1);

    cv::imshow("Integrated Optic System", output);

    // --- Encode frame to JPEG and push to shared buffer for HTTP server ---
    std::vector<uchar> jpegBuf;
    cv::imencode(".jpg", output, jpegBuf,
      { cv::IMWRITE_JPEG_QUALITY, 85 });
    {
      std::lock_guard<std::mutex> lock(g_frame.mtx);
      g_frame.jpegBuf = std::move(jpegBuf);
      g_frame.ready = true;
    }

    // --- Keyboard control --------------------------------------------------
    char key = (char)cv::waitKey(30);
    if (key == 27) break;
    if (key == 'o' || key == 'O') {
      currentMode = MODE_ORIGINAL;
      std::cout << "[MODE] Normal + Gesture" << std::endl;
    }
    if (key == 't' || key == 'T') {
      currentMode = MODE_THERMAL;
      std::cout << "[MODE] Thermal" << std::endl;
    }
    if (key == 'n' || key == 'N') {
      currentMode = MODE_NIGHTVISION;
      std::cout << "[MODE] NVG + Targeting" << std::endl;
    }
    if (key == '1' && activeStream != 1) {
      cap.release();
      cap.open(STREAM_1, cv::CAP_FFMPEG);
      if (cap.isOpened()) {
        activeStream = 1;
        std::cout << "[STREAM] Switched to stream 1: " << STREAM_1 << std::endl;
      }
      else {
        std::cerr << "[ERROR] Cannot open stream 1: " << STREAM_1 << std::endl;
      }
    }
    if (key == '2' && activeStream != 2) {
      cap.release();
      cap.open(STREAM_2, cv::CAP_FFMPEG);
      if (cap.isOpened()) {
        activeStream = 2;
        std::cout << "[STREAM] Switched to stream 2: " << STREAM_2 << std::endl;
      }
      else {
        std::cerr << "[ERROR] Cannot open stream 2: " << STREAM_2 << std::endl;
      }
    }
  }

  g_running = false;
  serverThread.join();
  cap.release();
  cv::destroyAllWindows();
  return 0;
}