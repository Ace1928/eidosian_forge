import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
def filter_audio_paths(path: str, language: str, lst_name: str):
    """Extract audio paths for the given language."""
    audio_paths = []
    path = Path(path)
    with open(path / 'scoring' / lst_name) as f:
        for line in f:
            audio_path, lang = line.strip().split()
            if language is not None and lang != language:
                continue
            audio_path = re.sub('^.*?\\/', '', audio_path)
            audio_paths.append(path / audio_path)
    return audio_paths