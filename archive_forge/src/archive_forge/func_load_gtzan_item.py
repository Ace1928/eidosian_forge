import os
from pathlib import Path
from typing import Optional, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
def load_gtzan_item(fileid: str, path: str, ext_audio: str) -> Tuple[Tensor, str]:
    """
    Loads a file from the dataset and returns the raw waveform
    as a Torch Tensor, its sample rate as an integer, and its
    genre as a string.
    """
    label, _ = fileid.split('.')
    file_audio = os.path.join(path, label, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)
    return (waveform, sample_rate, label)