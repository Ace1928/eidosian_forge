import numbers
import os
import sys
import warnings
from configparser import ConfigParser
from operator import itemgetter
from pathlib import Path
from typing import (
from scrapy.exceptions import ScrapyDeprecationWarning, UsageError
from scrapy.settings import BaseSettings
from scrapy.utils.deprecate import update_classpath
from scrapy.utils.python import without_none_values
def closest_scrapy_cfg(path: Union[str, os.PathLike]='.', prevpath: Optional[Union[str, os.PathLike]]=None) -> str:
    """Return the path to the closest scrapy.cfg file by traversing the current
    directory and its parents
    """
    if prevpath is not None and str(path) == str(prevpath):
        return ''
    path = Path(path).resolve()
    cfgfile = path / 'scrapy.cfg'
    if cfgfile.exists():
        return str(cfgfile)
    return closest_scrapy_cfg(path.parent, path)