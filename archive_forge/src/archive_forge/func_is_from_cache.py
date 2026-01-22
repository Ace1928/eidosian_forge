import os
from contextlib import contextmanager
from datetime import datetime
from typing import BinaryIO, Generator, Optional, Union
from pip._vendor.cachecontrol.cache import SeparateBodyBaseCache
from pip._vendor.cachecontrol.caches import SeparateBodyFileCache
from pip._vendor.requests.models import Response
from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import ensure_dir
def is_from_cache(response: Response) -> bool:
    return getattr(response, 'from_cache', False)