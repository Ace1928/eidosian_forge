import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from huggingface_hub.hf_api import list_models
from torch import nn
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
def save_tokenizer_config(dest_dir: Path, separate_vocabs=False):
    dname = dest_dir.name.split('-')
    dct = {'target_lang': dname[-1], 'source_lang': '-'.join(dname[:-1]), 'separate_vocabs': separate_vocabs}
    save_json(dct, dest_dir / 'tokenizer_config.json')