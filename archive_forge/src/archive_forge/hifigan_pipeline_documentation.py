from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.nn import Module
from torchaudio._internal import load_state_dict_from_url
from torchaudio.prototype.models.hifi_gan import hifigan_vocoder, HiFiGANVocoder
from torchaudio.transforms import MelSpectrogram
Generate mel spectrogram from a waveform. Should have same sample rate as ``self.sample_rate``.

        Args:
            waveform (Tensor): waveform of shape ``(batch_size, time_length)``.
        Returns:
            Tensor of shape ``(batch_size, n_mel, time_length)``
        