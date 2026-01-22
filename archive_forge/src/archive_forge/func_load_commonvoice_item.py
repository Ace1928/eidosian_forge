import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
def load_commonvoice_item(line: List[str], header: List[str], path: str, folder_audio: str, ext_audio: str) -> Tuple[Tensor, int, Dict[str, str]]:
    if header[1] != 'path':
        raise ValueError(f"expect `header[1]` to be 'path', but got {header[1]}")
    fileid = line[1]
    filename = os.path.join(path, folder_audio, fileid)
    if not filename.endswith(ext_audio):
        filename += ext_audio
    waveform, sample_rate = torchaudio.load(filename)
    dic = dict(zip(header, line))
    return (waveform, sample_rate, dic)