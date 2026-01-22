import sys
import warnings
from contextlib import contextmanager
from typing import Any, Generator, Union
from urllib.parse import urlsplit
import requests
from urllib3.exceptions import InsecureRequestWarning
import sphinx
from sphinx.config import Config
@contextmanager
def ignore_insecure_warning(**kwargs: Any) -> Generator[None, None, None]:
    with warnings.catch_warnings():
        if not kwargs.get('verify') and InsecureRequestWarning:
            warnings.filterwarnings('ignore', category=InsecureRequestWarning)
        yield