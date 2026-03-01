# Tactical Optical System & MJPEG Server

A multi-threaded C++ computer vision application utilizing OpenCV and Windows Sockets (`WinSock2`). This system ingests remote IP camera feeds, applies real-time military-grade optical simulations or hand-gesture recognition, and broadcasts the processed footage via a built-in HTTP MJPEG server.

## 🚀 Key Features
* **Built-in HTTP Server:** Runs a dedicated background thread to stream the processed video feed (MJPEG) to any web browser.
* **Network Stream Ingestion:** Dynamically connects to external IP cameras via FFmpeg, with real-time stream switching.
* **Gesture Recognition Mode:** Utilizes HSV skin masking, Convex Hulls, and Convexity Defects to mathematically classify hand gestures (Fist, Peace, Open Hand, etc.).
* **Thermal Vision Simulation:** Maps grayscale intensity to heat signatures (`COLORMAP_INFERNO`) and automatically tracks/counts hotspots.
* **Night Vision (NVG) & Targeting:** Applies CLAHE contrast enhancement, green-channel filtering, synthetic noise, and Haar Cascades for automated face (target) acquisition in low light.

## ⚙️ Prerequisites
* C++17 or higher
* OpenCV (built with FFmpeg support for network streams)
* Windows OS (uses `WinSock2` for the server architecture)

## 🛠️ Setup & Installation
1. Ensure `haarcascade_frontalface_default.xml` is placed in the working directory.
2. Link `Ws2_32.lib` and OpenCV libraries during compilation.
3. Run the executable.
4. Open a web browser and navigate to the specified local/public IP and port (default: `http://localhost:8082`) to view the live broadcast.

## 🎮 Controls
Use the following keyboard shortcuts in the local preview window to switch modes:

| Key | Action |
| :---: | :--- |
| **`O`** | **Normal Mode:** Standard feed with Hand Gesture & Face Detection. |
| **`T`** | **Thermal Mode:** Heat-signature simulation and hotspot tracking. |
| **`N`** | **Night Vision Mode:** NVG filter + Automated Target Acquisition. |
| **`1`** | **Stream 1:** Switch input to the primary network camera. |
| **`2`** | **Stream 2:** Switch input to the secondary network camera. |
| **`ESC`** | **Exit:** Safely shuts down the server threads and closes the app. |
