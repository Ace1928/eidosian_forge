import csv
import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
def load_cmuarctic_item(line: str, path: str, folder_audio: str, ext_audio: str) -> Tuple[Tensor, int, str, str]:
    utterance_id, transcript = line[0].strip().split(' ', 2)[1:]
    transcript = transcript[1:-3]
    file_audio = os.path.join(path, folder_audio, utterance_id + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)
    return (waveform, sample_rate, transcript, utterance_id.split('_')[1])