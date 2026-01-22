import warnings
from typing import List, Optional, Union
import numpy
import numpy as np
from ...audio_utils import mel_filter_bank, spectrogram
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import (
def mel_spectrogram(self, sequence: np.ndarray):
    """
        Generates MelSpectrogram.

        Args:
            sequence (`numpy.ndarray`):
                The sequence of which the mel-spectrogram will be computed.
        """
    mel_specs = []
    for seq in sequence:
        window = np.hanning(self.window_size + 1)[:-1]
        mel_specs.append(spectrogram(waveform=seq, window=window, frame_length=self.window_size, hop_length=self.hop_length, power=2.0, mel_filters=self.mel_filters))
    mel_specs = np.array(mel_specs)
    return mel_specs