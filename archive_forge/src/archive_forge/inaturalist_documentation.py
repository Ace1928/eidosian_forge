import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset

        Args:
            category_type(str): one of "full", "kingdom", "phylum", "class", "order", "family", "genus" or "super"
            category_id(int): an index (class id) from this category

        Returns:
            the name of the category
        