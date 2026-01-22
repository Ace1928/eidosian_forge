import warnings
from typing import List, Optional, Union
import numpy
import numpy as np
from ...audio_utils import mel_filter_bank, spectrogram
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import (
def interpolate_beat_times(self, beat_times: numpy.ndarray, steps_per_beat: numpy.ndarray, n_extend: numpy.ndarray):
    """
        This method takes beat_times and then interpolates that using `scipy.interpolate.interp1d` and the output is
        then used to convert raw audio to log-mel-spectrogram.

        Args:
            beat_times (`numpy.ndarray`):
                beat_times is passed into `scipy.interpolate.interp1d` for processing.
            steps_per_beat (`int`):
                used as an parameter to control the interpolation.
            n_extend (`int`):
                used as an parameter to control the interpolation.
        """
    requires_backends(self, ['scipy'])
    beat_times_function = scipy.interpolate.interp1d(np.arange(beat_times.size), beat_times, bounds_error=False, fill_value='extrapolate')
    ext_beats = beat_times_function(np.linspace(0, beat_times.size + n_extend - 1, beat_times.size * steps_per_beat + n_extend))
    return ext_beats