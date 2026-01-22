import itertools
from .utils import db_to_float
def detect_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Returns a list of all silent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_nonsilent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    seg_len = len(audio_segment)
    if seg_len < min_silence_len:
        return []
    silence_thresh = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude
    silence_starts = []
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])
    for i in slice_starts:
        audio_slice = audio_segment[i:i + min_silence_len]
        if audio_slice.rms <= silence_thresh:
            silence_starts.append(i)
    if not silence_starts:
        return []
    silent_ranges = []
    prev_i = silence_starts.pop(0)
    current_range_start = prev_i
    for silence_start_i in silence_starts:
        continuous = silence_start_i == prev_i + seek_step
        silence_has_gap = silence_start_i > prev_i + min_silence_len
        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start, prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i
    silent_ranges.append([current_range_start, prev_i + min_silence_len])
    return silent_ranges