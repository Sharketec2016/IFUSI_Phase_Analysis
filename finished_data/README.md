# Multi-Camera Phase Analysis & Clock Drift Study

This project provides a specialized toolkit for measuring and analyzing sub-frame timing synchronization and temporal drift between multiple astronomical cameras. By utilizing an LED as a shared reference signal, the system quantifies the relative phase offsets and determines if camera clocks are drifting apart over time.

## 🎯 Project Objectives

1.  **Quantify Lag:** Measure the constant frame-offset between camera pairs with sub-frame precision.
2.  **Identify Clock Drift:** Detect if a camera's internal clock is running slightly faster or slower than others, causing the phase offset to change over the course of an observation.
3.  **Automate Calibration:** Provide a robust, reproducible pipeline for calibrating camera arrays using FITS data.

## 🛠️ Technical Methodology

The analysis employs a two-tier approach to signal processing:

### 1. Global FFT Cross-Correlation
To find the average offset for an entire data cube, the script performs cross-correlation in the frequency domain:
- **Normalization:** Signals are zero-meaned and normalized by their standard deviation.
- **Zero-Padding:** Linear correlation is ensured by padding signals to $2N$ length.
- **FFT Multiplication:** The cross-power spectrum is computed ($\text{Corr} = \mathcal{F}^{-1}\{ \mathcal{F}(A) \cdot \mathcal{F}(B)^* \}$).
- **Sub-frame Interpolation:** The peak location is refined using **Parabolic Interpolation** on the three points surrounding the maximum, achieving precision significantly better than a single frame.

### 2. Sliding Window Drift Analysis
To track how synchronization evolves over time, the script implements a sliding window correlation:
- **Windows:** The signal is divided into overlapping chunks (default: `WINDOW_SIZE = 60`, `STEP_SIZE = 10`).
- **Local Correlation:** FFT cross-correlation is performed within each window.
- **Linear Regression:** The resulting "lags vs. time" plot is fitted with a linear model ($y = ax + b$).
- **Drift Rate:** The slope ($a$) represents the clock drift in **frames per frame**. Calculated later into ms/frame

## 📊 Analysis Parameters

- **Frequency (`Hz`):** 10 and 5 Hz LED signal.
- **Exposure Time:** ~0.040s (effectively the frame rate).
- **Window Size:** 60 frames (6 seconds of data per "snapshot").
- **Step Size:** 10 frames (1-second resolution for drift tracking).

## 📂 Project Structure

- `fft_phase_analysis.py`: The core engine for signal processing, correlation, and drift fitting.
- `boxes.json`: Configuration file defining the Region of Interest (ROI) for the LED in each camera's field of view.
- `Data/`: Directory containing organized FITS cubes (Frames × Height × Width).
- `fft_phase_analysis/`: Output directory for generated plots, histograms, and the final CSV report.

## 📈 Interpretation of Results

- **Phase Offset:** The fundamental lag in frames between two data sets compared. 
- **Slope (Drift):** 
    - Represents tha phase drift over the 500 frames between the cameras. 
    - A positive slope indicates the "other" camera is lagging further behind over time.
    - A negative slope indicates the "other" camera is catching up or running faster.
- **Standard Error:** Calculated via the covariance matrix of the linear fit, providing a measure of the drift's reliability.

## Statistical Measurements
- **Phase Drift Probability** All phase drift (slope) values are binned into a histogram. Number of bins chosen to be $\sqrt(Number of slope points)$. Gaussian model is fit for estimation on phase drift distribution. Probability calculated that a particular phase drift exceeds $3\sigma$ from the model fit ($1 - P(|drift| > 3\sigma)$). Calculated by passing the calculated/observed phase drive values into a Normal CDF with parameters $\mu$ and $\sigma$ (`norm.cdf(3*sigma, loc=mu, scale=sigma)`), $P(X \leq x)$. Probability that a phase drift is LESS than the threshold. 

---
*Developed for SCSU Astro Lab*
