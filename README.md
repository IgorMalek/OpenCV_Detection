# Tactical Optical System Simulator

A real-time C++ computer vision project utilizing OpenCV to simulate military-grade optical systems. This application captures a live webcam feed and applies digital image processing techniques to simulate Thermal Imaging and Night Vision (NVG), complete with automated target acquisition using Haar Cascades.

## 🚀 Features
* **Real-Time Processing:** Optimized C++ pipeline for low-latency video manipulation.
* **Thermal Vision Simulation:** Utilizes OpenCV colormaps (`COLORMAP_INFERNO`) on grayscale matrices to simulate heat signatures.
* **Night Vision Mode (NVG):** Enhances low-light visibility using CLAHE (Contrast Limited Adaptive Histogram Equalization), applies a green-channel filter, and generates synthetic noise to mimic light-intensifier tubes.
* **Target Acquisition:** Integrates Haar Cascade classifiers to automatically detect and track human faces (targets) within the enhanced Night Vision feed.

## ⚙️ Prerequisites
* C++11 or higher
* OpenCV (version 4.x recommended)
* CMake (optional, for building)

## 🛠️ Setup & Installation
1. Clone this repository.
2. Ensure OpenCV is properly linked in your build environment (e.g., via `CMakeLists.txt`).
3. **Important:** Download the `haarcascade_frontalface_default.xml` file from the official OpenCV GitHub repository and place it in your project's working directory (the same folder as your `.exe` file).
4. Compile and run the application.

## 🎮 Controls
Use the following keyboard shortcuts to switch between vision modes in real-time:

| Key | Action |
| :---: | :--- |
| **`O`** | **Original Mode:** Standard RGB camera feed. |
| **`T`** | **Thermal Mode:** Switches to heat-signature simulation. |
| **`N`** | **Night Vision Mode:** Activates NVG filter + Target Acquisition (Face Detection). |
| **`ESC`** | **Exit:** Closes the application and releases the camera. |

## 💡 About
This project was developed as a portfolio piece to demonstrate proficiency in C++, memory management (`cv::Mat`), and applied computer vision techniques relevant to the defense and simulation (mil-tech) industries.
