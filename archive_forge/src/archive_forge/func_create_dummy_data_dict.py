import os
import re
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional, Union
from zipfile import ZipFile
from ..utils.file_utils import cached_path, hf_github_url
from ..utils.logging import get_logger
from ..utils.version import Version
def create_dummy_data_dict(self, path_to_dummy_data, data_url):
    dummy_data_dict = {}
    for key, single_urls in data_url.items():
        for download_callback in self.download_callbacks:
            if isinstance(single_urls, list):
                for single_url in single_urls:
                    download_callback(single_url)
            else:
                single_url = single_urls
                download_callback(single_url)
        if isinstance(single_urls, list):
            value = [os.path.join(path_to_dummy_data, urllib.parse.quote_plus(Path(x).name)) for x in single_urls]
        else:
            single_url = single_urls
            value = os.path.join(path_to_dummy_data, urllib.parse.quote_plus(Path(single_url).name))
        dummy_data_dict[key] = value
    if all((isinstance(i, str) for i in dummy_data_dict.values())) and len(set(dummy_data_dict.values())) < len(dummy_data_dict.values()):
        dummy_data_dict = {key: value + key for key, value in dummy_data_dict.items()}
    return dummy_data_dict