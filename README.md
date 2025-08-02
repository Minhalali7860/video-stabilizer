
# Python Video Stabilizer

A simple and efficient **video stabilization** tool written in Python using **OpenCV** and **NumPy**.  
It automatically detects camera shake, smooths the motion, and outputs a stabilized version of your video.

## Features
- Detects and tracks motion between frames
- Smooths the camera trajectory to remove jitter
- Supports multiple video formats (`.mp4`, `.mov`, `.mkv`, `.avi`)
- Processes all videos in the `raw_video` folder automatically
- Saves stabilized videos in the `stabilized_clips` folder

## Project Structure
├── raw_video/ # Place your original shaky videos here
├── stabilized_clips/ # Stabilized videos will be saved here
├── stabilizer.py # Main script
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## Installation

1. **Clone the repository**

git clone https://github.com/YOUR-USERNAME/python-video-stabilizer.git
cd python-video-stabilizer.

2. **Install dependencies**
pip install -r requirements.txt

3.**Create folders**
mkdir raw_video stabilized_clips

## Usage
1)Put your shaky videos into the raw_video folder.
2)Run the script:
  python stabilizer.py
3)The stabilized videos will appear in stabilized_clips.

## Example
Input: Shaky handheld video
Output: Smooth stabilized version

## How It Works
Feature Detection – Detects key points in each frame using cv2.goodFeaturesToTrack.

Optical Flow Tracking – Tracks feature points between consecutive frames.

Affine Transformation – Estimates motion and extracts translation & rotation.

Trajectory Smoothing – Applies a moving average to smooth the motion path.

Frame Warping – Adjusts frames to match the smoothed trajectory.


## License
This project is licensed under the MIT License - feel free to use and modify it.
