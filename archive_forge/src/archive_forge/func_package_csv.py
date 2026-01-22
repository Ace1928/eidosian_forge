from __future__ import annotations # isort:skip
import hashlib
import json
from os.path import splitext
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, Any, TextIO
from urllib.parse import urljoin
from urllib.request import urlopen
def package_csv(module: str, name: str, **kw: Any) -> pd.DataFrame:
    import pandas as pd
    return pd.read_csv(package_path(name), **kw)