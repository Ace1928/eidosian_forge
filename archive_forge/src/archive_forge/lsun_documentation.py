import io
import os.path
import pickle
import string
from collections.abc import Iterable
from typing import Any, Callable, cast, List, Optional, Tuple, Union
from PIL import Image
from .utils import iterable_to_str, verify_str_arg
from .vision import VisionDataset

        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        