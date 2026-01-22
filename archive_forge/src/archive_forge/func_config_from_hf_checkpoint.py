import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from transformers import GPT2Config, LlamaConfig
from einops import rearrange
def config_from_hf_checkpoint(checkpoint_path: Union[str, os.PathLike], model_name: str) -> LlamaConfig:
    return LlamaConfig.from_pretrained(Path(checkpoint_path) / f'{model_name}-hf' / 'config.json')