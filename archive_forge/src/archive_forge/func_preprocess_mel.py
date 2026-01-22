import warnings
from typing import List, Optional, Union
import numpy
import numpy as np
from ...audio_utils import mel_filter_bank, spectrogram
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import (
def preprocess_mel(self, audio: np.ndarray, beatstep: np.ndarray):
    """
        Preprocessing for log-mel-spectrogram

        Args:
            audio (`numpy.ndarray` of shape `(audio_length, )` ):
                Raw audio waveform to be processed.
            beatstep (`numpy.ndarray`):
                Interpolated values of the raw audio. If beatstep[0] is greater than 0.0, then it will be shifted by
                the value at beatstep[0].
        """
    if audio is not None and len(audio.shape) != 1:
        raise ValueError(f'Expected `audio` to be a single channel audio input of shape `(n, )` but found shape {audio.shape}.')
    if beatstep[0] > 0.0:
        beatstep = beatstep - beatstep[0]
    num_steps = self.num_bars * 4
    num_target_steps = len(beatstep)
    extrapolated_beatstep = self.interpolate_beat_times(beat_times=beatstep, steps_per_beat=1, n_extend=(self.num_bars + 1) * 4 + 1)
    sample_indices = []
    max_feature_length = 0
    for i in range(0, num_target_steps, num_steps):
        start_idx = i
        end_idx = min(i + num_steps, num_target_steps)
        start_sample = int(extrapolated_beatstep[start_idx] * self.sampling_rate)
        end_sample = int(extrapolated_beatstep[end_idx] * self.sampling_rate)
        sample_indices.append((start_sample, end_sample))
        max_feature_length = max(max_feature_length, end_sample - start_sample)
    padded_batch = []
    for start_sample, end_sample in sample_indices:
        feature = audio[start_sample:end_sample]
        padded_feature = np.pad(feature, ((0, max_feature_length - feature.shape[0]),), 'constant', constant_values=0)
        padded_batch.append(padded_feature)
    padded_batch = np.asarray(padded_batch)
    return (padded_batch, extrapolated_beatstep)