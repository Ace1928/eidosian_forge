from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def download_pretrained_files(model: str) -> _PretrainedFiles:
    """
    Retrieves pretrained data files used for :func:`ctc_decoder`.

    Args:
        model (str): pretrained language model to download.
            Valid values are: ``"librispeech-3-gram"``, ``"librispeech-4-gram"`` and ``"librispeech"``.

    Returns:
        Object with the following attributes

            * ``lm``: path corresponding to downloaded language model,
              or ``None`` if the model is not associated with an lm
            * ``lexicon``: path corresponding to downloaded lexicon file
            * ``tokens``: path corresponding to downloaded tokens file
    """
    files = _get_filenames(model)
    lexicon_file = download_asset(files.lexicon)
    tokens_file = download_asset(files.tokens)
    if files.lm is not None:
        lm_file = download_asset(files.lm)
    else:
        lm_file = None
    return _PretrainedFiles(lexicon=lexicon_file, tokens=tokens_file, lm=lm_file)