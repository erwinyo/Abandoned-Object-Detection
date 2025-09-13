# Abandoned Object Detection (AOD)
Detect and classify abandoned object on public space.

[![forthebadge](https://forthebadge.com/images/featured/featured-built-with-love.svg)](https://forthebadge.com)

## Description
Abandoned Object Detection (AOD) systems are designed to automatically identify unattended or suspicious objects in public areas using computer vision and machine learning techniques. These systems **help enhance public safety** by providing real-time alerts and supporting security personnel in monitoring large spaces efficiently.

## Table of Content
- [Demo](#demo)
- [Stack](#stack)
- [Limitation](#limitation)

## Demo
[![Watch the demo](https://img.youtube.com/vi/yKqzroT37xY/0.jpg)](https://youtu.be/yKqzroT37xY)

## Stack
- **Video and Frame Processing**: OpenCV
- **Classify Object**: YOLOv11-classification

## Installation
**1. Clone The Repository**
```Bash
git clone http://github.com/Abandoned-Object-Detection
cd Abandoned-Object-Detection
```

**2. Install Packages**
```Bash
pip install -r requirements.txt
```

> **Note:**  
> The current project has been tested on Windows 11 + CUDA 11.8 installed with python version 3.10.11. If you have different version of CUDA, please refer to [pytorch installation](https://pytorch.org/get-started/locally/) or [previous versions](https://pytorch.org/get-started/previous-versions/) 

**3. Video Configuration**
```python
video_choosen = "video3"
poly_used = POLY_ZONE_VIDEO3
file_path = "videos//video3.avi"
still_bg_path = "videos//video3.png"
```
> **Variables:**  
> - **video_choosen**: Folder name where the output path.
> - **poly_used**: Preset polygon coordinates defining the area of interest in the video. (You can create your own with [PolygonZone from Roboflow](https://polygonzone.roboflow.com/))
> - **file_path**: Path to the input video file.
> - **still_bg_path**: Path to the reference background image used for comparison.

**4. Run Program**
```based
python main.py
```

## Limitation
1. The algorithm does not adjust for changes in room lighting. This means the background image (ground truth) and the video stream must have the same lighting condition.
2. **Good result:** Video1, Video2, Video3, Video6-cutted, Video7-cutted, Video8-cutted, Video9, and Video10
3. **Bad result:** Video4, Video5
4. **Not included (have dynamic lighting condition)**: Video6, Video7, Video8

***-cutted:** mean that the video has been cutted and trim to the scene that has good lighting condition







