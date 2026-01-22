import warnings
from typing import Optional, Union
import numpy as np
def get_mel_filter_banks(nb_frequency_bins: int, nb_mel_filters: int, frequency_min: float, frequency_max: float, sample_rate: int, norm: Optional[str]=None, mel_scale: str='htk') -> np.array:
    warnings.warn('The function `get_mel_filter_banks` is deprecated and will be removed in version 4.31.0 of Transformers', FutureWarning)
    return mel_filter_bank(num_frequency_bins=nb_frequency_bins, num_mel_filters=nb_mel_filters, min_frequency=frequency_min, max_frequency=frequency_max, sampling_rate=sample_rate, norm=norm, mel_scale=mel_scale)