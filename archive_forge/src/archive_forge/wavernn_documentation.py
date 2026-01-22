import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
Inference method of WaveRNN.

        This function currently only supports multinomial sampling, which assumes the
        network is trained on cross entropy loss.

        Args:
            specgram (Tensor):
                Batch of spectrograms. Shape: `(n_batch, n_freq, n_time)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``specgram`` contains spectrograms with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The inferred waveform of size `(n_batch, 1, n_time)`.
                1 stands for a single channel.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        