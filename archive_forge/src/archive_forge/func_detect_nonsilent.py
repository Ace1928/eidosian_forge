import itertools
from .utils import db_to_float
def detect_nonsilent(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Returns a list of all nonsilent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_silent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step)
    len_seg = len(audio_segment)
    if not silent_ranges:
        return [[0, len_seg]]
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []
    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i
    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])
    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)
    return nonsilent_ranges