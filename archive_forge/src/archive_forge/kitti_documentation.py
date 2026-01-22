import csv
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive
from .vision import VisionDataset
Download the KITTI data if it doesn't exist already.