from typing import Any, Dict, List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
def generate_noise(self, noise_length: int, generator: Optional[np.random.Generator]=None) -> np.ndarray:
    """
        Generates a random noise sequence of standard Gaussian noise for use in the `noise_sequence` argument of
        [`UnivNetModel.forward`].

        Args:
            spectrogram_length (`int`):
                The length (dim 0) of the generated noise.
            model_in_channels (`int`, *optional*, defaults to `None`):
                The number of features (dim 1) of the generated noise. This should correspond to the
                `model_in_channels` of the [`UnivNetGan`] model. If not set, this will default to
                `self.config.model_in_channels`.
            generator (`numpy.random.Generator`, *optional*, defaults to `None`)
                An optional `numpy.random.Generator` random number generator to control noise generation. If not set, a
                new generator with fresh entropy will be created.

        Returns:
            `numpy.ndarray`: Array containing random standard Gaussian noise of shape `(noise_length,
            model_in_channels)`.
        """
    if generator is None:
        generator = np.random.default_rng()
    noise_shape = (noise_length, self.model_in_channels)
    noise = generator.standard_normal(noise_shape, dtype=np.float32)
    return noise