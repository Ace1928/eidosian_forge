import functools
from typing import Callable, Generator, Iterable, Iterator, Optional, Tuple
from pip._vendor.rich.progress import (
from pip._internal.utils.logging import get_indentation
def get_download_progress_renderer(*, bar_type: str, size: Optional[int]=None) -> DownloadProgressRenderer:
    """Get an object that can be used to render the download progress.

    Returns a callable, that takes an iterable to "wrap".
    """
    if bar_type == 'on':
        return functools.partial(_rich_progress_bar, bar_type=bar_type, size=size)
    else:
        return iter