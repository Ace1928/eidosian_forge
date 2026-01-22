import json
import shlex
import subprocess
from typing import Tuple
import torch
def get_hvalue(self, weight):
    return weight.data.storage().data_ptr()