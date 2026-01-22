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
def check_equal(marian_cfg, k1, k2):
    v1, v2 = (marian_cfg[k1], marian_cfg[k2])
    if v1 != v2:
        raise ValueError(f'hparams {k1},{k2} differ: {v1} != {v2}')