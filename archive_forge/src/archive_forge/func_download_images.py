import os
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from .folder import default_loader
from .utils import check_integrity, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def download_images(self) -> None:
    if path.exists(self.images_dir):
        raise RuntimeError(f'The directory {self.images_dir} already exists. If you want to re-download or re-extract the images, delete the directory.')
    file, md5 = self._IMAGES_META[self.split, self.small]
    download_and_extract_archive(urljoin(self._BASE_URL, file), self.root, md5=md5)
    if self.split.startswith('train'):
        os.rename(self.images_dir.rsplit('_', 1)[0], self.images_dir)