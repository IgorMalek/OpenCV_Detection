#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <string>

enum VisionMode {
  MODE_ORIGINAL,
  MODE_THERMAL,
  MODE_NIGHTVISION
};

const int HOT_THRESHOLD = 220;
const double MIN_AREA = 500.0;
const double HAND_MIN_AREA = 6000.0;

std::string classifyGesture(const std::vector<cv::Point>& contour,
  const std::vector<cv::Vec4i>& defects,
  const cv::Rect& bbox)
{
  if (defects.empty()) return "FIST";

  int fingerCount = 0;
  double contourArea = cv::contourArea(contour);

  for (const auto& defect : defects) {
    cv::Point start = contour[defect[0]];
    cv::Point end = contour[defect[1]];
    cv::Point far = contour[defect[2]];
    float depth = defect[3] / 256.0f;

    if (depth < 20.0f) continue;

    double a = cv::norm(end - start);
    double b = cv::norm(far - start);
    double c = cv::norm(end - far);
    double angle = std::acos((b * b + c * c - a * a) / (2 * b * c)) * 180.0 / CV_PI;

    if (angle < 90.0) fingerCount++;
  }

  float aspectRatio = (float)bbox.width / (float)bbox.height;

  if (fingerCount == 0) return "FIST";
  if (fingerCount == 1) return "PEACE / POINT";
  if (fingerCount == 2) return "THREE FINGERS";
  if (fingerCount == 3) return "FOUR FINGERS";
  if (fingerCount >= 4) return "OPEN HAND";

  return "UNKNOWN";
}

void processModeNormal(const cv::Mat& frame, cv::Mat& output, cv::CascadeClassifier& faceCascade)
{
  cv::Mat hsv, skinMask, blurred, gray;

  cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
  cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);

  cv::inRange(hsv,
    cv::Scalar(0, 20, 75),
    cv::Scalar(20, 255, 255),
    skinMask);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, kernel);
  cv::GaussianBlur(skinMask, skinMask, cv::Size(3, 3), 0);

  std::vector<cv::Rect> faces;
  faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(60, 60));
  for (const auto& face : faces) {
    int pad = (int)(face.width * 0.15);
    cv::Rect expanded(
      std::max(0, face.x - pad),
      std::max(0, face.y - pad),
      std::min(skinMask.cols - face.x + pad, face.width + 2 * pad),
      std::min(skinMask.rows - face.y + pad, face.height + 2 * pad)
    );
    skinMask(expanded) = 0;
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  for (const auto& face : faces) {
    cv::rectangle(output, face, cv::Scalar(255, 100, 0), 2);
    cv::putText(output, "FACE",
      cv::Point(face.x, face.y - 8),
      cv::FONT_HERSHEY_SIMPLEX, 0.55,
      cv::Scalar(255, 100, 0), 1);
  }

  int handCount = 0;

  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area < HAND_MIN_AREA) continue;

    cv::Rect bboxCheck = cv::boundingRect(contour);
    float aspectRatio = (float)bboxCheck.width / (float)bboxCheck.height;
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
      cv::FONT_HERSHEY_SIMPLEX, 0.65,
      cv::Scalar(0, 255, 0), 2);

    std::string areaStr = "AREA: " + std::to_string((int)area) + "px";
    cv::putText(output, areaStr,
      cv::Point(bbox.x, bbox.y + bbox.height + 18),
      cv::FONT_HERSHEY_SIMPLEX, 0.45,
      cv::Scalar(200, 200, 200), 1);
  }

  std::string handStr = "HANDS DETECTED: " + std::to_string(handCount);
  cv::putText(output, handStr,
    cv::Point(10, 55),
    cv::FONT_HERSHEY_SIMPLEX, 0.6,
    cv::Scalar(0, 255, 255), 2);

  cv::putText(output, "MODE: NORMAL | GESTURE DETECTION",
    cv::Point(10, 30),
    cv::FONT_HERSHEY_SIMPLEX, 0.65,
    cv::Scalar(0, 255, 0), 2);
}

int main()
{
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "Camera Error!" << std::endl;
    return -1;
  }

  cv::CascadeClassifier faceCascade;
  if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
    std::cerr << "Error: File not found haarcascade_frontalface_default.xml!" << std::endl;
    return -1;
  }

  cv::Mat frame, output;
  VisionMode currentMode = MODE_ORIGINAL;

  std::cout << "Optic system ready." << std::endl;
  std::cout << " [O] Normal + Gesture | [T] Thermal | [N] Nightvision + Targeting | [ESC] Exit" << std::endl;

  while (true) {
    cap >> frame;
    if (frame.empty()) break;

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
          cv::FONT_HERSHEY_SIMPLEX, 0.45,
          cv::Scalar(0, 255, 0), 1);
      }

      std::string countStr = "TARGETS: " + std::to_string(targetCount);
      cv::putText(thermal, countStr,
        cv::Point(10, 55),
        cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(0, 255, 255), 2);

      cv::putText(thermal, "MODE: THERMAL",
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

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
          cv::FONT_HERSHEY_SIMPLEX, 0.5,
          cv::Scalar(0, 0, 255), 1);
      }

      cv::putText(output, "MODE: NVG + TARGETING",
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(0, 255, 0), 2);
    }
    // ===================== MODE: NORMAL =====================
    else {
      processModeNormal(frame, output, faceCascade);
    }

    cv::imshow("Integrated Optic System", output);

    char key = (char)cv::waitKey(30);
    if (key == 27) break;
    if (key == 'o' || key == 'O') currentMode = MODE_ORIGINAL;
    if (key == 't' || key == 'T') currentMode = MODE_THERMAL;
    if (key == 'n' || key == 'N') currentMode = MODE_NIGHTVISION;
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}