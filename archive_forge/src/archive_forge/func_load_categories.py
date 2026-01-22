import os
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from .folder import default_loader
from .utils import check_integrity, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def load_categories(self, download: bool=True) -> Tuple[List[str], Dict[str, int]]:

    def process(line: str) -> Tuple[str, int]:
        cls, idx = line.split()
        return (cls, int(idx))
    file, md5 = self._CATEGORIES_META
    file = path.join(self.root, file)
    if not self._check_integrity(file, md5, download):
        self.download_devkit()
    with open(file) as fh:
        class_to_idx = dict((process(line) for line in fh))
    return (sorted(class_to_idx.keys()), class_to_idx)