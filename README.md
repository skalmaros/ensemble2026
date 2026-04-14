## ECG Image-to-Signal Digitization Pipeline
Overview
This project provides a complete end-to-end system for converting analog ECG scans (paper printouts) into high-fidelity digital 1D time-series signals. The system combines Deep Learning (image segmentation) with advanced signal processing to accurately extract ECG traces and calibrate them to real-world medical units (mV, ms).

This is the solution to Task 4 for the EnsebleAI 2026 Hackathon. The goal was to build a pipeline that takes a photo/scan of a paper ECG as input and digitizes it back into a 1D time-series signal.


# Key Features
1. Signal Segmentation (U-Net Model)
The core of the system is a U-Net neural network with a ResNet34 encoder. This model is trained to separate the actual ECG trace from background noise, millimeter grid paper, printed text, and scanning artifacts.

2. Intelligent Layout Analysis
 - The system automatically parses the physical structure of the ECG page:
 - Detects horizontal baselines (isoelectric lines) using histogram analysis.
 - Dynamically determines column and row boundaries, splitting a standard 12-lead printout into individual segments (e.g., I, II, III, aVR, aVL, aVF, V1-V6).

3. Precise Digitization & Centroid Method
 Instead of simple pixel sampling, the algorithm calculates the mathematical centroid for each vertical column of pixels. This allows for:
  - Generating a perfectly thin signal line (1 px), even if the model's mask is thick or jagged.
  - Applying linear interpolation to seamlessly fill any gaps or missing data points in the mask.

4. Physical Calibration and Resampling
 The system operates on physical units rather than pixels, adhering strictly to medical standards:
 - Y-axis (Voltage): Scales pixels to millivolts (standard: 10 mm/mV).
 - X-axis (Time): Scales horizontal width to time based on standard paper feed speed (25 mm/s).
 - Resampling: The final signal is resampled to a frequency of 500 Hz, which is the standard in clinical electrocardiography.

# Input & Output
 - Input: Raw scanned ECG images in PNG, JPG, or JPEG formats.
 - Output: A compressed .npz file containing 1D NumPy arrays for all 12 leads, saved in float16 precision to optimize storage space and memory usage.

# Technology Stack
 - PyTorch (U-Net modeling and inference)
 - Segmentation Models Pytorch (SMP) (Encoder architecture)
 - OpenCV (Image processing and grid visualization)
 - SciPy & NumPy (Interpolation, peak detection, and matrix operations)
 - Matplotlib (Generating comparative dashboards)
 # Quick Start
 ```bash
 # Install required dependencies
pip install torch segmentation-models-pytorch opencv-python scipy numpy wfdb

# Run the digitization pipeline
python viterbi.py --input data/test --model unet_best.pth
```
