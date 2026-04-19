import time

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from astropy.io import fits
import pandas as pd
from scipy.optimize import curve_fit
from itertools import combinations


WINDOWS_SIZE = 60           #Window size for sliding window correlation (in frames)
STEP_SIZE = 10              #Step size for sliding window correlation (in frames)
HZ = 5                     #Frequency of LED
EXPOSURE_TIME = 0.040       #units of seconds


#the effective readout rate is negligible because of frame transfer. 
# So our exposure time is effectively the frame time, which is 1/Hz. 
# So we can use this to estimate how many frames correspond to one period of the signal, which can help us 
#       choose appropriate window sizes for the sliding window correlation.

PERIOD = 1 / HZ             #Period of the LED signal in seconds

boxes_filepath = "./boxes.json"



#================================================================================
"""
Helper functions for FFT-based phase analysis of camera signals.
"""

"""
Linear model for curve fitting Phase Drift (Clock Drift) over time.
"""
def linear_model(x, a, b):
    return a * x + b

"""
Lists files in a directory, excluding subdirectories. 
If the directory does not exist, it raises an error.
"""
def list_files_classic(directory_path):
    '''
    List all files in a directory, excluding subdirectories.'''
    # Get all entries in the directory
    entries = os.listdir(directory_path)
    
    # Filter out directories to keep only files
    files = [f for f in entries if os.path.isfile(os.path.join(directory_path, f))]
    
    return files

"""
Load in FITS data from a given file path and return the data array.
"""
def load_data(file_path):
    hdul = fits.open(file_path)
    return hdul[0].data

"""
Calculate mean intensities within a specified box region for each frame in the FITS file.
"""
def cal_intensities(file, cam_id, box):
    data = load_data(file)
    xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']
    intensities = np.mean(data[:, ymin:ymax+1, xmin:xmax+1], axis=(1, 2))

    return intensities

"""
Confirm that a folder path exists; if it doesn't, create it
"""
def confirm_folder_path(folder_path):
    """Ensure that the folder exists; create it if it doesn't."""
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            pass
    else:
        os.makedirs(folder_path, exist_ok=True)
        print(f"The folder '{folder_path}' has been created.")
 
"""
Perform a sub frame estimation of cross correltaion peak.
"""
def subframe_peak_location(correlation):
    """
    Refines peak location using parabolic interpolation
    to achieve sub-frame accuracy.

    Parabolic interpolation is used because the peaks from the cross correltaion are shaped
    like a parabola. By fitting a parabola to the peak and its immediate neighbors,
    we can estimate the true peak location.
    """
    peak = np.argmax(correlation)

    # Ensure peak is not at edge
    if 1 <= peak < len(correlation) - 1:

        y0 = correlation[peak - 1]
        y1 = correlation[peak]
        y2 = correlation[peak + 1]

        denom = 2 * (2*y1 - y0 - y2)

        if denom != 0:
            delta = (y0 - y2) / denom
            peak = peak + delta

    return peak

def sliding_window_correlation(main_sig, other_sig, window_size=60, step=10, main_camera_id=None, other_camera_id=None, plot=True, save_path=None, file_name=None):
    """
    Performs cross-correlation in chunks to see if lag changes over time.
    """
    lags = []
    time_indices = []
    
    # Normalize the full signals first to save computation
    a_full = (main_sig - np.mean(main_sig)) / np.std(main_sig)
    b_full = (other_sig - np.mean(other_sig)) / np.std(other_sig)

    # Slide the window across the data
    for start in range(0, len(a_full) - window_size, step):
        end = start + window_size
        
        # Extract the window
        a_win = a_full[start:end]
        b_win = b_full[start:end]
        
        # Standard FFT Correlation on this specific window
        Nfft = 2 * len(a_win)
        A = np.fft.rfft(a_win, n=Nfft)
        B = np.fft.rfft(b_win, n=Nfft)
        corr = np.fft.irfft(A * np.conj(B), n=Nfft).real
        
        # Find the peak
        raw_lag = np.argmax(corr)
        # Correct for wraparound (negative lags)
        if raw_lag > len(a_win):
            raw_lag -= Nfft
            
        lags.append(raw_lag)
        time_indices.append(start)
        
    # if plot and main_camera_id is not None and other_camera_id is not None and save_path is not None and file_name is not None:

    #     fig3 = plt.figure(figsize=(10, 4))
    #     plt.plot(time_indices, lags, 'o-')
    #     plt.title("Intra-file Clock Drift Analysis")
    #     plt.xlabel("Frame Number (Start of Window)")
    #     plt.ylabel("Measured Lag (Frames)")
    #     plt.grid(True)
    #     plt.savefig(f"{save_path}/{main_camera_id}vs{other_camera_id}/Intra-file_Clock_Drift__{file_name}.png")
    #     # plt.show()
    #     plt.close(fig3)


    return time_indices, lags

def fft_cross_correlation(main_intensities, other_intensities, main_camera_id=None, other_camera_id=None, plot=True, save_path=None, file_name=None):

    # a = a[:300]
    # b = b[:300]

    # Normalize signals
    a = (main_intensities - np.mean(main_intensities)) / np.std(main_intensities)
    b = (other_intensities - np.mean(other_intensities)) / np.std(other_intensities)


    if plot and main_camera_id is not None and other_camera_id is not None and save_path is not None and file_name is not None:
        fig1 = plt.figure(figsize=(10, 5))
        plt.title("Normalized Intensity Signals of Camera {} and Camera {}".format(main_camera_id, other_camera_id))
        plt.plot(a, label=f'{main_camera_id}')
        plt.plot(b, label=f'{other_camera_id}')
        plt.xlabel("Frame Index")
        plt.ylabel("Normalized Intensity")
        plt.legend()

        confirm_folder_path(f"{save_path}/{main_camera_id}vs{other_camera_id}")
        plt.savefig(f"{save_path}/{main_camera_id}vs{other_camera_id}/normalized_signals_cam__{file_name}.png", dpi=300)
        
        
        # plt.show()
        plt.close(fig1)


    N = len(a)

    # Zero-padding for linear correlation
    Nfft = 2 * N

    A = np.fft.rfft(a, n=Nfft)
    B = np.fft.rfft(b, n=Nfft)

    cross_power = A * np.conj(B)

    correlation = np.fft.irfft(cross_power,n=Nfft).real

    max_peak = subframe_peak_location(correlation)

    if max_peak > N:
        max_peak -= Nfft

    phase_offset = max_peak


    if plot and main_camera_id is not None and other_camera_id is not None and save_path is not None and file_name is not None:
        fig2 = plt.figure(figsize=(10, 5))
        plt.title("Cross-Correlation between Camera {} and Camera {}".format(main_camera_id, other_camera_id))
        plt.plot(correlation)
        plt.vlines(np.argmax(correlation), 0, correlation[np.argmax(correlation)], color='r', label='Peak')
        plt.xlabel("Lag (frames)")
        plt.ylabel("Cross-Correlation")
        plt.legend()
        confirm_folder_path(f"{save_path}/{main_camera_id}vs{other_camera_id}")
        plt.savefig(f"{save_path}/{main_camera_id}vs{other_camera_id}/Cross-Correlation__{file_name}.png", dpi=300)

        # plt.show()
        plt.close(fig2)

    return max_peak, phase_offset

def plot_phase_drift_histogram(phase_df):
    grouped_sets = phase_df.groupby(["Main Camera", "Other Camera"])

    for (main_cam, other_cam), df in grouped_sets:

        bins = int(np.floor(np.sqrt(len(df))))
        
        # Phase Offset Histogram
        fig5 = plt.figure(figsize=(10, 5))

        plt.hist(
            df["Phase Offset"],
            bins=bins,
            edgecolor='black'
        )

        plt.title(
            f"Histogram of Phase Offsets between Camera {main_cam} and Camera {other_cam}"
        )

        plt.xlabel("Phase Offset (frames)")
        plt.ylabel("Frequency")

        plt.savefig(
            f"./fft_phase_analysis/{Hz}Hz/"
            f"phase_offset_histogram_cam_{main_cam}_vs_{other_cam}.png",
            dpi=300

        )

        plt.close(fig5)

        # Clock Drift Histogram
        fig6 = plt.figure(figsize=(10, 5))

        plt.hist(
            df["Slope"],
            bins=bins,
            edgecolor='black'
        )

        plt.title(
            f"Histogram of Clock Drift Slopes between Camera {main_cam} and Camera {other_cam}"
        )

        plt.xlabel("Clock Drift Slope (frames/frame)")
        plt.ylabel("Frequency")

        plt.savefig(
            f"./fft_phase_analysis/{Hz}Hz/"
            f"clock_drift_slope_histogram_cam_{main_cam}_vs_{other_cam}.png",
            dpi=300
        )

        plt.close(fig6)

def plot_drift_fit(indices, lags, popt,
                   main_camera_id,
                   other_camera_id,
                   save_path,
                   file_name):

    fitted_line = linear_model(np.array(indices), *popt)

    fig7 = plt.figure(figsize=(10,5))

    plt.plot(indices, lags, 'o', label="Measured Lag")
    plt.plot(indices, fitted_line, '-', label="Linear Fit")

    plt.title(
        f"Clock Drift Fit: Camera {main_camera_id} vs {other_camera_id}"
    )

    plt.xlabel("Frame Index (Window Start)")
    plt.ylabel("Lag (frames)")

    plt.legend()
    plt.grid(True)

    confirm_folder_path(
        f"{save_path}/{main_camera_id}vs{other_camera_id}"
    )

    text_str = f"Slope: {popt[0]:.4f} frames/frame\nIntercept: {popt[1]:.2f} frames"
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(
        f"{save_path}/{main_camera_id}vs{other_camera_id}/drift_fit__{file_name}.png",
        dpi=300
    )

    plt.close(fig7)


def gather_intensities(main_camera_path, other_camera_path, main_box, other_box, main_camera, other_cam_id, file):
    main_intensities = cal_intensities(
        os.path.join(main_camera_path, file),
        main_camera,
        main_box
    )

    other_intensities = cal_intensities(
        os.path.join(other_camera_path, other_file),
        other_cam_id,
        other_box
    )

    return main_intensities, other_intensities


def curve_fit_phase_drift(indices, lags, plot_data):
    popt, pcov = curve_fit(linear_model, indices, lags)
    a_fit, b_fit = popt
    a_err, b_err = np.sqrt(np.diag(pcov))
    
    if plot_data:
        plot_drift_fit(
            indices, lags, popt,
            main_camera_id=main_camera,
            other_camera_id=other_cam_id,
            save_path=f"./fft_phase_analysis/{Hz}Hz",
            file_name=f"{file.split('.')[0]}"
        )
    return popt, pcov


if __name__ == "__main__":

    camera_ids = [12574, 12606, 13251, 13703]
    Hz = HZ

    boxes = json.load(open(boxes_filepath, "r"))

    camera_pairs = list(combinations(camera_ids, 2))

    total_files_to_process = len(camera_pairs) * 15
    processed_sofar = 0

    phase_df = pd.DataFrame()

    results_list = []
    start_time = time.time()

    for main_camera, other_cam_id in camera_pairs:


        main_camera_path = f"./Data/{main_camera}/3D_CUBE/{Hz}Hz"
        other_camera_path = f"./Data/{other_cam_id}/3D_CUBE/{Hz}Hz"

        main_camera_files = list_files_classic(main_camera_path)
        other_camera_files = list_files_classic(other_camera_path)

        main_box = boxes[f"{Hz}Hz"][str(main_camera)]
        other_box = boxes[f"{Hz}Hz"][str(other_cam_id)]

        plot_data = True

        for file in main_camera_files:
            for other_file in other_camera_files:


                if file != other_file:
                    continue


                processed_sofar += 1
                print(f"Files left to process: {total_files_to_process - processed_sofar - 1}")



                main_intensities, other_intensities = gather_intensities(
                    main_camera_path, other_camera_path,
                    main_box, other_box,
                    main_camera, other_cam_id,
                    file
                )

                max_peak, phase_offset = fft_cross_correlation(
                    main_intensities,
                    other_intensities,
                    main_camera_id=main_camera,
                    other_camera_id=other_cam_id,
                    plot=plot_data,
                    save_path=f"./fft_phase_analysis/{Hz}Hz",
                    file_name=f"{file.split('.')[0]}"
                )

                indices, lags = sliding_window_correlation(
                    main_intensities,
                    other_intensities,
                    window_size=WINDOWS_SIZE,
                    step=STEP_SIZE,
                    main_camera_id=main_camera,
                    other_camera_id=other_cam_id,
                    plot=plot_data,
                    save_path=f"./fft_phase_analysis/{Hz}Hz",
                    file_name=f"{file.split('.')[0]}"
                )

                popt, pcov = curve_fit_phase_drift(indices, lags, plot_data)
                a_fit, b_fit = popt
                a_err, b_err = np.sqrt(np.diag(pcov))

                time_offset = phase_offset * EXPOSURE_TIME
                phase_deg = (360 * time_offset / PERIOD)

                results_list.append({
                    "Main Camera": main_camera,
                    "Other Camera": other_cam_id,
                    "Max Peak": float(max_peak),
                    "Phase Offset": int(phase_offset),
                    "Phase Degrees": float(phase_deg),
                    "Time Offset (s)": float(time_offset),
                    "Window Size": WINDOWS_SIZE,
                    "Step Size": STEP_SIZE,
                    "Slope": float(a_fit),
                    "Intercept": float(b_fit),
                    "Slope Error": float(a_err),
                    "Intercept Error": float(b_err),
                    "File": f"{file}"
                })

                plot_data = False



                    

        new_df = pd.DataFrame(results_list)

        phase_df = pd.concat([phase_df, new_df], ignore_index=True)
        results_list = []
        

    phase_df.to_csv(f"./fft_phase_analysis/{Hz}Hz/phase_analysis_results.csv", index=False)
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    
    print("All done!")





