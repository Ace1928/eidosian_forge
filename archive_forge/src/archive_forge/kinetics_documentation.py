import csv
import os
import time
import urllib
from functools import partial
from multiprocessing import Pool
from os import path
from typing import Any, Callable, Dict, Optional, Tuple
from torch import Tensor
from .folder import find_classes, make_dataset
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .video_utils import VideoClips
from .vision import VisionDataset
move videos from
        split_folder/
            ├── clip1.avi
            ├── clip2.avi

        to the correct format as described below:
        split_folder/
            ├── class1
            │   ├── clip1.avi

        