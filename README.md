# FSD-APAM: Frequency-Spatial Joint Decoupling with Adaptive Perceptual Aggregation for Remote Sensing Small Object Detection

This is the official implementation of the paper **"Frequency-Spatial Joint Decoupling with Adaptive Perceptual Aggregation for Remote Sensing Small Object Detection"**. 

## 💡 Introduction
Small object detection in remote sensing imagery is extremely challenging due to limited spatial resolution and complex backgrounds. FSD-APAM is a novel **plug-and-play** framework rooted in decoupled representation learning, designed to rescue weak target features from annihilation during downsampling.

### Key Components:
* **FDD (Frequency-domain Detail Decoupling):** Reconstructs high-frequency edge energy using Discrete Wavelet Transform (DWT).
* **SSD (Spatial Salience Decoupling):** Simulates biological lateral inhibition to isolate weak singularities from background interference.
* **ACPA (Adaptive Contextual Perceptual Aggregation):** Connects sparse keypoints and dense semantic grids with linear complexity ($O(HW)$) for global awareness.
