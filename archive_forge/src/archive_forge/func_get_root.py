import torch
import numpy as np
import argparse
from typing import Dict
def get_root(x, dependency_map):
    if x in dependency_map:
        return get_root(dependency_map[x], dependency_map)
    else:
        return x