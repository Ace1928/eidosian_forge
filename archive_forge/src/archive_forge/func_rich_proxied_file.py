import io
from typing import IO, TYPE_CHECKING, Any, List
from .ansi import AnsiDecoder
from .text import Text
@property
def rich_proxied_file(self) -> IO[str]:
    """Get proxied file."""
    return self.__file