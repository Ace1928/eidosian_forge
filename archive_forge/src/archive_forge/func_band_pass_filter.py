from scipy.signal import butter, sosfilt
from .utils import (register_pydub_effect,stereo_to_ms,ms_to_stereo)
@register_pydub_effect
def band_pass_filter(seg, low_cutoff_freq, high_cutoff_freq, order=5):
    filter_fn = _mk_butter_filter([low_cutoff_freq, high_cutoff_freq], 'band', order=order)
    return seg.apply_mono_filter_to_each_channel(filter_fn)