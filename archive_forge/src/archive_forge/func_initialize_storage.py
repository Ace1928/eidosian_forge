from __future__ import annotations
from urllib.parse import quote
import tornado.web
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorageError
from streamlit.runtime.memory_media_file_storage import (
from streamlit.web.server import allow_cross_origin_requests
@classmethod
def initialize_storage(cls, storage: MemoryMediaFileStorage) -> None:
    """Set the MemoryMediaFileStorage object used by instances of this
        handler. Must be called on server startup.
        """
    cls._storage = storage