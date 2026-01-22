import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union, cast
import pgzip
import torch
from torch import Tensor
from fairscale.internal.containers import from_np, to_np
from .utils import ExitCode

        Update the reference count.

        If the reference counting file does not have this sha1, then a new tracking
        entry of the added.

        Args:
            current_sha1_hash (str):
                The sha1 hash of the incoming added file.
            inc (bool):
                Increment or decrement.

        Returns:
            (int):
                Resulting ref count.
        