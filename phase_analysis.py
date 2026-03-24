import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ── classify every frame in a cube ───────────────────────────────────────────
'''
For each camera, we have a 3D array of frames (shape: n_frames x H x W).
We also have a bounding box (xmin,xmax,ymin,ymax) that contains the LED, and
lists of known ON and OFF frames.

Using the known ON frames, we compute the average pixel intensity in the bounding box
to get a threshold.  Then we classify every frame as ON (1) or OFF (0) based on
whether the average intensity in the box is above or below that threshold.


'''
def classify_frames(df, ON_FRAMES, OFF_FRAMES,
                    xmin, xmax, ymin, ymax,
                    start_index=10):
    avg_led_on = []
    #Compute the average pixel intensity in the bounding box for each known ON frame
    for i in ON_FRAMES:
        sub = df[i][ymin:ymax+1, xmin:xmax+1]
        avg_led_on.append(np.mean(sub))

    # Set the threshold as the minimum of these average ON intensities
    threshold = np.min(avg_led_on)
    n = df.shape[0]
    states = np.full(n, np.nan)
    # Classify each frame as ON (1) or OFF (0) based on the average intensity in the box
    new_start_index = np.max(ON_FRAMES) + 1;
    for i in range(new_start_index, n):
        sub = df[i][ymin:ymax+1, xmin:xmax+1]
        states[i] = 1.0 if np.mean(sub) > threshold else 0.0


    for index in range(start_index, new_start_index, 1):
        if(index in ON_FRAMES):
            states[index] = 1.0
        else:
            states[index] = 0.0
    



    return states


# ── find every transition index ───────────────────────────────────────────────

def find_transitions(states, start_index=10):
    rising, falling = [], []
    valid = np.where(~np.isnan(states))[0]
    valid = valid[valid >= start_index]

    for k in range(1, len(valid)):
        prev_i = valid[k-1]
        curr_i = valid[k]
        if curr_i - prev_i > 1:
            continue
        if states[prev_i] == 0 and states[curr_i] == 1:
            rising.append(curr_i)
        elif states[prev_i] == 1 and states[curr_i] == 0:
            falling.append(curr_i)

    return rising, falling


# ── phase delay: one reference camera vs all others ──────────────────────────

def phase_delay_analysis(camera_configs, start_index=10, reference_cam=None):
    if reference_cam is None:
        reference_cam = camera_configs[0]['name']

    results = {}
    for cfg in camera_configs:
        name = cfg['name']
        states = classify_frames(
            cfg['df'], cfg['ON_FRAMES'], cfg['OFF_FRAMES'],
            cfg['xmin'], cfg['xmax'], cfg['ymin'], cfg['ymax'],
            start_index=start_index
        )
        rising, falling = find_transitions(states, start_index=start_index)
        results[name] = {'states': states, 'rising': rising, 'falling': falling}
        print(f"Camera {name}: {len(rising)} rising edges, "
              f"{len(falling)} falling edges")

    cam_names = [cfg['name'] for cfg in camera_configs]
    phase_table = []

    for edge_type in ('rising', 'falling'):
        ref_edges = results[reference_cam][edge_type]
        cam_pointers = {name: 0 for name in cam_names if name != reference_cam}

        for cycle, ref_frame in enumerate(ref_edges):
            row = {
                'edge':                   edge_type,
                'cycle':                  cycle,
                'ref_cam':                reference_cam,
                'ref_frame':              ref_frame,
                reference_cam:            ref_frame,
                f'{reference_cam}_phase': 0,
            }

            for name in cam_names:
                if name == reference_cam:
                    continue
                edges = results[name][edge_type]
                ptr   = cam_pointers[name]

                while ptr < len(edges) and edges[ptr] < ref_frame:
                    ptr += 1
                cam_pointers[name] = ptr

                if ptr >= len(edges):
                    row[name]            = None
                    row[f'{name}_phase'] = None
                else:
                    cam_frame            = edges[ptr]
                    row[name]            = cam_frame
                    row[f'{name}_phase'] = cam_frame - ref_frame
                    cam_pointers[name]   = ptr + 1

            phase_table.append(row)

    return results, phase_table


# ── all-pairs phase matrix ────────────────────────────────────────────────────

def all_pairs_phase(camera_configs, start_index=10):
    cam_names = [cfg['name'] for cfg in camera_configs]
    all_tables = {}

    for ref in cam_names:
        _, pt = phase_delay_analysis(camera_configs,
                                     start_index=start_index,
                                     reference_cam=ref)
        all_tables[ref] = pt

    pair_stats = {}
    for i, cam_a in enumerate(cam_names):
        for cam_b in cam_names[i+1:]:
            stats = {}
            for edge in ('rising', 'falling'):
                vals_ab = [r[f'{cam_b}_phase']
                           for r in all_tables[cam_a]
                           if r['edge'] == edge
                           and r.get(f'{cam_b}_phase') is not None]
                vals_ba = [-r[f'{cam_a}_phase']
                           for r in all_tables[cam_b]
                           if r['edge'] == edge
                           and r.get(f'{cam_a}_phase') is not None]
                all_vals = vals_ab + vals_ba
                stats[edge] = {
                    'mean': np.mean(all_vals) if all_vals else None,
                    'std':  np.std(all_vals)  if all_vals else None,
                    'n':    len(all_vals),
                }

            combined = []
            for edge in ('rising', 'falling'):
                vals = [r[f'{cam_b}_phase']
                        for r in all_tables[cam_a]
                        if r['edge'] == edge
                        and r.get(f'{cam_b}_phase') is not None]
                combined.extend(vals)
            pair_stats[(cam_a, cam_b)] = {
                **stats,
                'combined': {
                    'mean': np.mean(combined) if combined else None,
                    'std':  np.std(combined)  if combined else None,
                    'n':    len(combined),
                }
            }

    return pair_stats, all_tables


# ── print phase table ─────────────────────────────────────────────────────────

def print_phase_table(phase_table, camera_names):
    ref = phase_table[0]['ref_cam']
    others = [c for c in camera_names if c != ref]

    print(f"\n{'='*80}")
    print(f"  Phase delays  —  reference camera: {ref}")
    print(f"  Positive = other camera fires AFTER {ref}")
    print(f"{'='*80}")

    header = f"{'edge':>8}  {'cycle':>5}  {'ref_frame':>10}"
    for name in others:
        header += f"  {str(name):>8}  {'Δ':>5}"
    print(header)
    print('-' * len(header))

    for row in phase_table:
        line = (f"{row['edge']:>8}  {row['cycle']:>5}  "
                f"{row['ref_frame']:>10}")
        for name in others:
            frame = row.get(name)
            phase = row.get(f'{name}_phase')
            f_str = f"{frame:>8}" if frame is not None else f"{'N/A':>8}"
            p_str = f"{phase:>+5}" if phase is not None else f"{'N/A':>5}"
            line += f"  {f_str}  {p_str}"
        print(line)
    print()

    print(f"\n{'─'*50}")
    print("  Summary per camera per edge type")
    print(f"{'─'*50}")
    for name in others:
        for edge in ('rising', 'falling'):
            vals = [r[f'{name}_phase'] for r in phase_table
                    if r['edge'] == edge
                    and r.get(f'{name}_phase') is not None]
            if vals:
                print(f"  Cam {name}  {edge:>8}:  "
                      f"mean={np.mean(vals):+.2f}  std={np.std(vals):.2f}  "
                      f"min={np.min(vals):+d}  max={np.max(vals):+d}  "
                      f"(n={len(vals)})")
    print()


def print_all_pairs(pair_stats):
    print(f"\n{'='*80}")
    print("  All-pairs phase summary")
    print("  Interpretation: cam_B fires  mean  frames AFTER cam_A")
    print(f"{'='*80}")
    header = f"  {'cam_A':>8}  {'cam_B':>8}  {'edge':>8}  "
    header += f"{'mean':>8}  {'std':>6}  {'n':>5}"
    print(header)
    print('-' * len(header))
    for (cam_a, cam_b), stats in pair_stats.items():
        for edge in ('rising', 'falling', 'combined'):
            s = stats[edge]
            if s['mean'] is not None:
                print(f"  {str(cam_a):>8}  {str(cam_b):>8}  {edge:>8}  "
                      f"{s['mean']:>+8.2f}  {s['std']:>6.2f}  {s['n']:>5}")
    print()


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_phase_delays(phase_table, camera_names):
    """
    One subplot per non-reference camera.
    Title and filename both include the reference camera name so runs
    with different references are clearly distinguished and never overwrite.
    """
    ref = phase_table[0]['ref_cam']
    others = [c for c in camera_names if c != ref]

    fig, axes = plt.subplots(len(others), 1,
                             figsize=(14, 4 * len(others)),
                             sharex=True)
    if len(others) == 1:
        axes = [axes]

    markers = {'rising': 'o', 'falling': '^'}
    colors  = {'rising': 'steelblue', 'falling': 'tomato'}

    for ax, cam in zip(axes, others):
        for edge in ('rising', 'falling'):
            xs, ys = [], []
            for row in phase_table:
                if row['edge'] != edge:
                    continue
                phase = row.get(f'{cam}_phase')
                if phase is None:
                    continue
                xs.append(row['ref_frame'])
                ys.append(phase)
            if not xs:
                continue
            mean_y, std_y = np.mean(ys), np.std(ys)
            ax.scatter(xs, ys, marker=markers[edge], color=colors[edge],
                       label=f'{edge}  μ={mean_y:+.2f}  σ={std_y:.2f}',
                       s=50, zorder=3, alpha=0.85)
            ax.axhline(mean_y, color=colors[edge], lw=1.2, ls='--', alpha=0.5)

        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_ylabel('Phase delay [frames]')
        ax.set_title(f'Camera {cam} vs reference {ref}  '
                     f'(positive = cam {cam} lags ref {ref})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(f'Reference frame index  (camera {ref})')
    fig.suptitle(f'Phase Delay vs. Frame Index  —  reference: cam {ref}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f'phase_delays_ref{ref}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    # plt.show()
    print(f"Plot saved -> {fname}")


def plot_all_pairs(pair_stats, camera_names, ref):
    """
    Heatmap of mean combined phase between every pair of cameras.
    This is reference-agnostic (uses all-pairs data), so only one copy
    is ever needed — filename is fixed.
    """
    n = len(camera_names)
    matrix = np.full((n, n), np.nan)
    name_to_idx = {name: i for i, name in enumerate(camera_names)}

    for (cam_a, cam_b), stats in pair_stats.items():
        i = name_to_idx[cam_a]
        j = name_to_idx[cam_b]
        val = stats['combined']['mean']
        if val is not None:
            matrix[i, j] =  val
            matrix[j, i] = -val

    np.fill_diagonal(matrix, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Phase delay [frames]  (row fires before col)')

    labels = [str(c) for c in camera_names]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_title('All-pairs mean phase delay\n'
                 '(positive = column camera lags row camera)')

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                        fontsize=10,
                        color='white' if abs(val) > vmax * 0.6 else 'black')

    plt.tight_layout()
    plt.savefig(f'{ref}__phase_matrix.png', dpi=150, bbox_inches='tight')
    # plt.show()
    print("Plot saved -> phase_matrix.png")


def plot_led_states(results, start_index=10):
    names = list(results.keys())
    fig, axes = plt.subplots(len(names), 1,
                             figsize=(16, 2.5 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        states = results[name]['states']
        xs = np.arange(len(states))
        valid = ~np.isnan(states)
        ax.fill_between(xs[valid], states[valid],
                        step='mid', alpha=0.6, color='steelblue')
        ax.set_ylabel(f'Cam {name}')
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['OFF', 'ON'])
        ax.axvline(start_index, color='red', ls=':', lw=0.8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Frame index')
    fig.suptitle('Classified LED states per camera', fontsize=13)
    plt.tight_layout()
    plt.savefig('led_states.png', dpi=150, bbox_inches='tight')
    # plt.show()
    print("Plot saved -> led_states.png")



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          ADVANCED PHASE ALGORITHMS                                          ║
# ║                                                                              ║
# ║  The basic algorithm above measures phase in whole integer frames because   ║
# ║  it simply records WHICH frame index a transition occurred in.  But the     ║
# ║  true physical phase between cameras is almost certainly a non-integer —    ║
# ║  the cameras started at slightly different sub-frame times.                 ║
# ║                                                                              ║
# ║  Two better approaches are implemented below:                               ║
# ║                                                                              ║
# ║  1. SUB-FRAME INTERPOLATION  — look at the pixel intensity values around    ║
# ║     each transition and fit a line to find the exact fractional frame       ║
# ║     where the LED crossed the threshold.  Precision: ~0.1 frame.            ║
# ║                                                                              ║
# ║  2. FFT CROSS-CORRELATION    — treat the entire ON/OFF waveform as a        ║
# ║     signal and measure the lag between two cameras' signals in one shot     ║
# ║     using the Fourier transform.  Precision: ~0.01 frame.                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: SUB-FRAME INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────
#
# CONCEPT:
#   The basic algorithm says "the transition happened at frame N" because that
#   is where the state flips from 0→1 or 1→0.  But the actual LED brightness
#   doesn't jump instantaneously — it ramps up or down over the exposure.  By
#   looking at the raw pixel intensities in the frames around a transition we
#   can estimate the exact fractional frame where the brightness crossed the
#   threshold, e.g. "the transition happened at frame 47.3".
#
# HOW IT WORKS (step by step):
#   1. For each integer transition at frame N, take a small window of frames
#      around it:  [N - half_window, ..., N-1, N, N+1, ..., N + half_window]
#   2. Compute the mean pixel intensity in the bounding box for each frame
#      in that window — this gives us a small intensity time-series.
#   3. Fit a straight line (np.polyfit degree=1) through those intensities.
#      Near a rising edge the intensities go from low→high; near a falling
#      edge they go from high→low.  A line is a good local approximation.
#   4. Solve for where the line crosses the threshold:
#          threshold = slope * t + intercept
#          t_cross   = (threshold - intercept) / slope
#      t_cross is now a FRACTIONAL frame index.
#   5. Phase = t_cross(cam_B) - t_cross(cam_A)
#
# LIMITATIONS:
#   - Assumes the transition is roughly linear locally (good approximation).
#   - Noisy data will make the fit less precise; using a wider window helps
#     but risks including non-linear parts of the ramp.
#   - Still anchored to integer-frame transitions found by the basic algorithm.

def subframe_transition_time(df, xmin, xmax, ymin, ymax,
                              integer_transition, threshold,
                              half_window=3):
    """
    Given a detected integer transition at `integer_transition`,
    fit a line to the intensity values in a window around it and
    return the fractional frame index where intensity crosses threshold.

    Parameters
    ----------
    df                 : 3D array (frames x H x W)
    xmin,xmax,ymin,ymax: bounding box
    integer_transition : the integer frame index where the transition was detected
    threshold          : the ON/OFF threshold (min of known-ON averages)
    half_window        : how many frames either side of the transition to include
                         in the linear fit.  3 means we use 7 frames total.

    Returns
    -------
    float : fractional frame index of the threshold crossing,
            or float(integer_transition) if the fit fails.
    """
    n_frames = df.shape[0]

    # Build the window of frame indices, clamped to valid range
    t_start = max(0, integer_transition - half_window)
    t_end   = min(n_frames - 1, integer_transition + half_window)
    t_indices = np.arange(t_start, t_end + 1)

    # Compute mean intensity in the bounding box for each frame in the window
    intensities = np.array([
        np.mean(df[i][ymin:ymax+1, xmin:xmax+1])
        for i in t_indices
    ])

    # Need at least 2 points to fit a line
    if len(t_indices) < 2:
        return float(integer_transition)

    # Fit a degree-1 polynomial (straight line):  intensity = slope*t + intercept
    # np.polyfit returns [slope, intercept]
    coeffs = np.polyfit(t_indices, intensities, deg=1)
    slope, intercept = coeffs

    # If the slope is essentially zero, the signal isn't changing — can't
    # interpolate meaningfully, so fall back to the integer estimate
    if abs(slope) < 1e-10:
        return float(integer_transition)

    # Solve for where the line crosses the threshold:
    #   threshold = slope * t_cross + intercept
    #   t_cross   = (threshold - intercept) / slope
    t_cross = (threshold - intercept) / slope

    # Sanity check: the crossing should be near our window.
    # If it's wildly outside (bad fit), fall back to integer.
    if t_cross < t_start - half_window or t_cross > t_end + half_window:
        return float(integer_transition)

    return t_cross


def subframe_phase_analysis(camera_configs, results, start_index=10,
                             reference_cam=None, half_window=3):
    """
    Runs the sub-frame interpolation phase measurement.

    For each camera, we already have integer transition indices from
    find_transitions().  For each transition we refine it to a fractional
    frame using subframe_transition_time(), then compute phase as the
    difference in fractional frame times between the reference and each
    other camera.

    Parameters
    ----------
    camera_configs : list of camera config dicts (same format as before)
    results        : output of phase_delay_analysis() — contains integer
                     transitions and states for each camera
    start_index    : ignore frames before this
    reference_cam  : reference camera name
    half_window    : window half-width for the linear fit (default 3 frames)

    Returns
    -------
    sf_phase_table : list of dicts, one per (edge_type, cycle), with keys:
                       'edge', 'cycle', 'ref_cam', 'ref_frame_int',
                       'ref_frame_subf',          ← fractional ref transition
                       for each cam: '{cam}_frame_subf', '{cam}_phase_subf'
    """
    if reference_cam is None:
        reference_cam = camera_configs[0]['name']

    cam_names = [cfg['name'] for cfg in camera_configs]

    # Pre-compute the threshold and a lookup from name→cfg for each camera
    cfg_by_name = {cfg['name']: cfg for cfg in camera_configs}

    def get_threshold(name):
        cfg = cfg_by_name[name]
        avg_on = [np.mean(cfg['df'][i][cfg['ymin']:cfg['ymax']+1,
                                       cfg['xmin']:cfg['xmax']+1])
                  for i in cfg['ON_FRAMES']]
        return np.min(avg_on)

    thresholds = {name: get_threshold(name) for name in cam_names}

    sf_phase_table = []

    for edge_type in ('rising', 'falling'):
        ref_int_edges = results[reference_cam][edge_type]
        cam_pointers  = {n: 0 for n in cam_names if n != reference_cam}

        for cycle, ref_int_frame in enumerate(ref_int_edges):

            # Refine the reference camera's transition to sub-frame precision
            cfg_ref = cfg_by_name[reference_cam]
            ref_subf = subframe_transition_time(
                cfg_ref['df'],
                cfg_ref['xmin'], cfg_ref['xmax'],
                cfg_ref['ymin'], cfg_ref['ymax'],
                ref_int_frame,
                thresholds[reference_cam],
                half_window=half_window
            )

            row = {
                'edge':            edge_type,
                'cycle':           cycle,
                'ref_cam':         reference_cam,
                'ref_frame_int':   ref_int_frame,   # integer (from basic algo)
                'ref_frame_subf':  ref_subf,        # fractional (interpolated)
                f'{reference_cam}_frame_subf': ref_subf,
                f'{reference_cam}_phase_subf': 0.0,
            }

            for name in cam_names:
                if name == reference_cam:
                    continue

                int_edges = results[name][edge_type]
                ptr = cam_pointers[name]

                # Advance past edges before the reference frame
                while ptr < len(int_edges) and int_edges[ptr] < ref_int_frame:
                    ptr += 1
                cam_pointers[name] = ptr

                if ptr >= len(int_edges):
                    row[f'{name}_frame_subf'] = None
                    row[f'{name}_phase_subf'] = None
                    continue

                cam_int_frame = int_edges[ptr]
                cfg_cam = cfg_by_name[name]

                # Refine this camera's transition to sub-frame precision
                cam_subf = subframe_transition_time(
                    cfg_cam['df'],
                    cfg_cam['xmin'], cfg_cam['xmax'],
                    cfg_cam['ymin'], cfg_cam['ymax'],
                    cam_int_frame,
                    thresholds[name],
                    half_window=half_window
                )

                row[f'{name}_frame_subf'] = cam_subf
                row[f'{name}_phase_subf'] = cam_subf - ref_subf  # fractional phase
                cam_pointers[name] = ptr + 1

            sf_phase_table.append(row)

    return sf_phase_table


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: FFT CROSS-CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
#
# CONCEPT:
#   Instead of looking at individual transitions one at a time, treat the
#   entire sequence of ON/OFF states as a SIGNAL (a square wave) and measure
#   the time lag between two cameras' signals all at once.
#
#   Cross-correlation answers the question: "if I slide signal B along the
#   time axis, at what offset does it match signal A best?"  The offset at
#   peak correlation is the phase lag between the two signals.
#
# HOW IT WORKS (step by step):
#   1. Take the classified state arrays for two cameras (arrays of 0s and 1s).
#      These are our "signals".
#   2. Compute the FFT (Fast Fourier Transform) of both signals.
#      The FFT converts the time-domain signal into its frequency components.
#   3. Multiply the FFT of signal A by the COMPLEX CONJUGATE of the FFT of
#      signal B.  In frequency space, multiplication corresponds to
#      correlation in time space — this is the cross-power spectrum.
#   4. Take the inverse FFT of that product.  The result is the
#      cross-correlation function: a curve whose x-axis is "lag in frames"
#      and whose y-axis is "how well the signals match at that lag".
#   5. Find the lag at which the cross-correlation is maximum.
#      That lag is the phase offset between the two cameras.
#   6. For sub-frame precision: instead of taking argmax (integer), fit a
#      parabola to the peak and find its vertex analytically.
#
# WHY FFT?
#   Computing cross-correlation naively (slide and multiply) takes O(N²) time.
#   The FFT method does the same thing in O(N log N) — much faster for long
#   signals.  For 500 frames it doesn't matter much, but it's good practice.
#
# LIMITATIONS:
#   - Assumes the phase is CONSTANT across the whole dataset.  If the phase
#     is slowly drifting (which is what you're investigating!), the FFT gives
#     you the average phase over all 500 frames, not per-cycle.
#   - The result is a single number per camera pair, not a time series.
#   - Best used as a cross-check against the per-cycle methods.

def parabolic_peak(arr, peak_idx):
    """
    Fit a parabola to three points around a peak and return the
    fractional index of the parabola's vertex.

    Given three consecutive values y0, y1, y2 at indices i-1, i, i+1,
    the vertex of the parabola through those points is at:
        delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
    So the refined peak is at  peak_idx + delta.

    This gives sub-sample precision at essentially zero extra cost.
    """
    if peak_idx <= 0 or peak_idx >= len(arr) - 1:
        return float(peak_idx)   # can't fit at the edges

    y0 = arr[peak_idx - 1]
    y1 = arr[peak_idx]
    y2 = arr[peak_idx + 1]

    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(peak_idx)  # flat peak, no refinement possible

    delta = 0.5 * (y0 - y2) / denom
    return float(peak_idx) + delta


def fft_phase_between(states_a, states_b):
    """
    Compute the phase lag of signal B relative to signal A using
    FFT cross-correlation.

    Returns
    -------
    lag_frames : float
        How many frames B lags behind A.
        Positive = B fires after A.  Negative = B fires before A.
    correlation : np.ndarray
        The full cross-correlation curve (useful for plotting/debugging).
    """
    # Work only on the valid (non-NaN) overlap region
    valid = ~(np.isnan(states_a) | np.isnan(states_b))
    a = states_a[valid].copy()
    b = states_b[valid].copy()
    N = len(a)

    if N < 10:
        return np.nan, np.array([])

    # Zero-mean the signals.
    # This removes the DC offset (the average ON fraction) which would
    # otherwise create a large peak at lag=0 regardless of true phase.
    a -= np.mean(a)
    b -= np.mean(b)

    # FFT of both signals.
    # np.fft.rfft is the real-input FFT — more efficient than the full FFT
    # since our signals are real-valued (not complex).
    A = np.fft.rfft(a, n=N)
    B = np.fft.rfft(b, n=N)

    # Cross-power spectrum: A * conj(B)
    # In time domain this is equivalent to: correlate(a, b)
    # The conjugate flips B in frequency space, which is like
    # reversing it in time — that's exactly what correlation needs.
    cross_power = A * np.conj(B)

    # Normalise so the peak correlation = 1.0 regardless of signal amplitude.
    # This makes it easier to compare across camera pairs.
    norm = np.abs(cross_power)
    norm[norm < 1e-12] = 1.0          # avoid divide-by-zero
    cross_power_norm = cross_power / norm

    # Inverse FFT → cross-correlation in time domain
    correlation = np.fft.irfft(cross_power_norm, n=N).real

    # The correlation array is circular (it wraps around).
    # Lag 0 is at index 0.
    # Positive lags (B is behind A) are at indices 1, 2, 3, ...
    # Negative lags (B is ahead of A) are at indices N-1, N-2, N-3, ...
    # We need to "unwrap" this so that negative lags show as negative numbers.
    # np.fft.fftshift does exactly that — it moves the zero-lag to the centre.
    corr_shifted = np.fft.fftshift(correlation)
    lags = np.arange(-N//2, N - N//2)   # lag axis in frames

    # Find the integer index of the peak correlation
    peak_idx_shifted = int(np.argmax(corr_shifted))

    # Refine to sub-frame using parabolic interpolation around the peak
    peak_subframe = parabolic_peak(corr_shifted, peak_idx_shifted)

    # Convert array index back to a lag in frames
    # lags[N//2] == 0, so: lag = peak_subframe - N//2
    lag_frames = peak_subframe - N // 2

    return lag_frames, corr_shifted, lags


def fft_phase_analysis(camera_configs, results, reference_cam=None):
    """
    Compute the FFT cross-correlation phase between the reference camera
    and every other camera, and also between all pairs.

    Returns
    -------
    fft_results : dict keyed by (cam_a, cam_b) ->
                    {'lag': float,           ← B lags A by this many frames
                     'correlation': array,   ← full correlation curve
                     'lags': array}          ← lag axis for plotting
    """
    if reference_cam is None:
        reference_cam = camera_configs[0]['name']

    cam_names = [cfg['name'] for cfg in camera_configs]
    fft_results = {}

    # Compare reference vs every other camera
    for name in cam_names:
        if name == reference_cam:
            continue

        states_ref = results[reference_cam]['states']
        states_cam = results[name]['states']

        lag, corr, lags = fft_phase_between(states_ref, states_cam)
        fft_results[(reference_cam, name)] = {
            'lag':         lag,      # positive = name lags reference
            'correlation': corr,
            'lags':        lags,
        }
        print(f"  FFT: cam {name} lags cam {reference_cam} by "
              f"{lag:+.3f} frames")

    return fft_results


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON PLOT: integer vs sub-frame vs FFT
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_comparison(phase_table, sf_phase_table, fft_results,
                           camera_names):
    """
    Side-by-side comparison of all three methods for every non-reference camera.

    Layout: one row per camera, three columns:
      Left   — phase over time (integer method, dots)
               overlaid with sub-frame method (crosses) so you can see
               how much the interpolation shifts the estimate
      Middle — histogram of phase values for both per-cycle methods,
               showing the distribution and how tight each estimate is
      Right  — FFT cross-correlation curve with the peak marked,
               showing how clean the correlation is
    """
    ref = phase_table[0]['ref_cam']
    others = [c for c in camera_names if c != ref]

    fig, axes = plt.subplots(len(others), 3,
                             figsize=(18, 5 * len(others)))
    if len(others) == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing works

    for row_idx, cam in enumerate(others):

        ax_time  = axes[row_idx, 0]
        ax_hist  = axes[row_idx, 1]
        ax_fft   = axes[row_idx, 2]

        # ── Left: phase over time, integer vs sub-frame ──────────────────────
        for edge, marker_int, marker_sf, color in [
            ('rising',  'o', '+', 'steelblue'),
            ('falling', 'v', 'x', 'tomato'),
        ]:
            # Integer method
            xs_int, ys_int = [], []
            for r in phase_table:
                if r['edge'] == edge and r.get(f'{cam}_phase') is not None:
                    xs_int.append(r['ref_frame'])
                    ys_int.append(r[f'{cam}_phase'])

            # Sub-frame method
            xs_sf, ys_sf = [], []
            for r in sf_phase_table:
                if r['edge'] == edge and r.get(f'{cam}_phase_subf') is not None:
                    xs_sf.append(r['ref_frame_int'])
                    ys_sf.append(r[f'{cam}_phase_subf'])

            if xs_int:
                ax_time.scatter(xs_int, ys_int, marker=marker_int,
                                color=color, s=40, alpha=0.6,
                                label=f'{edge} integer '
                                      f'μ={np.mean(ys_int):+.2f}')
            if xs_sf:
                ax_time.scatter(xs_sf, ys_sf, marker=marker_sf,
                                color=color, s=60, alpha=0.9,
                                label=f'{edge} sub-frame '
                                      f'μ={np.mean(ys_sf):+.3f}')

        # FFT result as a horizontal band (mean ± uncertainty)
        fft_key = (ref, cam)
        if fft_key in fft_results:
            fft_lag = fft_results[fft_key]['lag']
            ax_time.axhline(fft_lag, color='green', lw=2, ls='-.',
                            label=f'FFT  {fft_lag:+.3f}')

        ax_time.axhline(0, color='gray', lw=0.8, ls=':')
        ax_time.set_xlabel(f'Ref frame index (cam {ref})')
        ax_time.set_ylabel('Phase delay [frames]')
        ax_time.set_title(f'Cam {cam} vs {ref}  — phase over time')
        ax_time.legend(fontsize=7, loc='upper right')
        ax_time.grid(True, alpha=0.3)

        # ── Middle: histogram ─────────────────────────────────────────────────
        all_int = [r[f'{cam}_phase'] for r in phase_table
                   if r.get(f'{cam}_phase') is not None]
        all_sf  = [r[f'{cam}_phase_subf'] for r in sf_phase_table
                   if r.get(f'{cam}_phase_subf') is not None]

        if all_int:
            ax_hist.hist(all_int, bins=20, alpha=0.5, color='steelblue',
                         label=f'Integer  μ={np.mean(all_int):+.2f} '
                               f'σ={np.std(all_int):.2f}')
        if all_sf:
            ax_hist.hist(all_sf, bins=40, alpha=0.5, color='tomato',
                         label=f'Sub-frame μ={np.mean(all_sf):+.3f} '
                               f'σ={np.std(all_sf):.3f}')
        if fft_key in fft_results:
            ax_hist.axvline(fft_results[fft_key]['lag'],
                            color='green', lw=2, ls='-.', label='FFT')

        ax_hist.set_xlabel('Phase delay [frames]')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f'Cam {cam} — phase distribution')
        ax_hist.legend(fontsize=7)
        ax_hist.grid(True, alpha=0.3)

        # ── Right: FFT correlation curve ──────────────────────────────────────
        if fft_key in fft_results:
            corr = fft_results[fft_key]['correlation']
            lags = fft_results[fft_key]['lags']
            lag  = fft_results[fft_key]['lag']

            # Only show the central ±30 frames for clarity
            window = 30
            mask = np.abs(lags) <= window
            ax_fft.plot(lags[mask], corr[mask], color='purple', lw=1.5)
            ax_fft.axvline(lag, color='green', lw=2, ls='--',
                           label=f'Peak at {lag:+.3f} frames')
            ax_fft.axvline(0, color='gray', lw=0.8, ls=':')
            ax_fft.set_xlabel('Lag [frames]')
            ax_fft.set_ylabel('Normalised correlation')
            ax_fft.set_title(f'FFT cross-correlation\ncam {ref} vs cam {cam}')
            ax_fft.legend(fontsize=8)
            ax_fft.grid(True, alpha=0.3)
        else:
            ax_fft.text(0.5, 0.5, 'No FFT data', transform=ax_fft.transAxes,
                        ha='center', va='center')

    ref_name = phase_table[0]['ref_cam']
    fig.suptitle(f'Phase measurement method comparison  —  reference: cam {ref_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = f'phase_method_comparison_ref{ref_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved -> {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED MAIN  (append at bottom — call after existing analysis)
# ─────────────────────────────────────────────────────────────────────────────
#
# To use the advanced methods, add this block after your existing main():




# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    cam_12574 = fits.open("./Data/12574/2026_03_23__17_14_12_cube.fits")
    cam_12606 = fits.open("./Data/12606/2026_03_23__17_14_12_cube.fits")
    cam_13251 = fits.open("./Data/13251/2026_03_23__17_14_12_cube.fits")
    cam_13703 = fits.open("./Data/13703/2026_03_23__17_14_12_cube.fits")

    df_12574 = cam_12574[0].data
    df_12606 = cam_12606[0].data
    df_13251 = cam_13251[0].data
    df_13703 = cam_13703[0].data

    cam_12574_ON  = [0, 1, 2, 3, 8, 9, 10, 11, 12, 17, 18, 19, 20]
    cam_12574_OFF = [4, 5, 6, 7, 13, 14, 15, 16, 21, 22, 23, 24, 30]

    cam_12606_ON  = [0, 1, 6, 7, 8, 9, 10, 14,15, 16, 17, 18, 23]
    cam_12606_OFF = [2, 3, 4, 5, 11, 12, 13, 19, 20, 21, 22, 28, 29]

    cam_13251_ON  = [0, 1, 2, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 24]
    cam_13251_OFF = [3, 4, 5, 11, 12, 13, 14, 20, 21, 22, 23, 29, 30, 31]

    cam_13703_ON  = [0, 1, 2, 3, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20]
    cam_13703_OFF = [3, 4, 5, 6, 12, 13, 14, 15, 21, 22, 23, 29, 30]


    boxes = {
        12574: (390, 450, 490, 530),
        12606: (380, 430, 490, 530),
        13251: (240, 330, 470, 550),
        13703: (390, 435, 510, 550),
    }

    camera_configs = [
        dict(name=12574, df=df_12574,
             ON_FRAMES=cam_12574_ON, OFF_FRAMES=cam_12574_OFF,
             xmin=boxes[12574][0], xmax=boxes[12574][1],
             ymin=boxes[12574][2], ymax=boxes[12574][3]),
        dict(name=12606, df=df_12606,
             ON_FRAMES=cam_12606_ON, OFF_FRAMES=cam_12606_OFF,
             xmin=boxes[12606][0], xmax=boxes[12606][1],
             ymin=boxes[12606][2], ymax=boxes[12606][3]),
        dict(name=13251, df=df_13251,
             ON_FRAMES=cam_13251_ON, OFF_FRAMES=cam_13251_OFF,
             xmin=boxes[13251][0], xmax=boxes[13251][1],
             ymin=boxes[13251][2], ymax=boxes[13251][3]),
        dict(name=13703, df=df_13703,
             ON_FRAMES=cam_13703_ON, OFF_FRAMES=cam_13703_OFF,
             xmin=boxes[13703][0], xmax=boxes[13703][1],
             ymin=boxes[13703][2], ymax=boxes[13703][3]),
    ]

    START_INDEX  = 10
    camera_names = [cfg['name'] for cfg in camera_configs]

    # ── LED state plot (only need this once) ──
    results_any, _ = phase_delay_analysis(
        camera_configs, start_index=START_INDEX, reference_cam=camera_names[0]
    )
    # plot_led_states(results_any, start_index=START_INDEX)

    # ── per-reference phase delay plots ──
    for ref in camera_names:
        print(f"\n{'#'*60}")
        print(f"  Reference camera: {ref}")
        print(f"{'#'*60}")
        results, phase_table = phase_delay_analysis(
            camera_configs, start_index=START_INDEX, reference_cam=ref
        )
        print_phase_table(phase_table, camera_names)
        plot_phase_delays(phase_table, camera_names)

        # ── all-pairs definitive phase matrix (reference-agnostic, run once) ──
        print("\nComputing all-pairs phase matrix...")
        pair_stats, all_tables = all_pairs_phase(camera_configs, start_index=START_INDEX)
        print_all_pairs(pair_stats)
        plot_all_pairs(pair_stats, camera_names, ref)

    # for ref in camera_names:
    #     results, phase_table = phase_delay_analysis(
    #         camera_configs, start_index=START_INDEX, reference_cam=ref)

    #     # Sub-frame interpolation
    #     sf_table = subframe_phase_analysis(
    #         camera_configs, results,
    #         start_index=START_INDEX, reference_cam=ref, half_window=3)

    #     # FFT cross-correlation
    #     fft_res = fft_phase_analysis(camera_configs, results, reference_cam=ref)

    #     # Comparison plot
    #     plot_method_comparison(phase_table, sf_table, fft_res, camera_names)
