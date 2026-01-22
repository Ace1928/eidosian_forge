import collections
import numpy
import os
import torch
from safetensors.torch import serialize_file, load_file
import argparse
def rename_key(rename, name):
    for k, v in rename.items():
        if k in name:
            name = name.replace(k, v)
    return name