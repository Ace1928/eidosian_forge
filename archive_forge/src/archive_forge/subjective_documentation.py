from typing import Tuple
import torch
import torch.nn as nn
import torchaudio
Predict subjective evaluation metric score.

        Args:
            waveform (torch.Tensor): Input waveform for evaluation. Tensor with dimensions `(batch, time)`.
            reference (torch.Tensor): Non-matching clean reference. Tensor with dimensions `(batch, time_ref)`.

        Returns:
            (torch.Tensor): Subjective metric score. Tensor with dimensions `(batch,)`.
        