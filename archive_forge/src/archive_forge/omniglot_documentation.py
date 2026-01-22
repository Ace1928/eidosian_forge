from os.path import join
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, list_dir, list_files
from .vision import VisionDataset

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        