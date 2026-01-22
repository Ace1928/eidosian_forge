from __future__ import annotations
from collections import abc
from typing import Any, Mapping, Optional, cast
from bson.objectid import ObjectId
from gridfs.errors import NoFile
from gridfs.grid_file import (
from pymongo import ASCENDING, DESCENDING, _csot
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.common import validate_string
from pymongo.database import Database
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import _ServerMode
from pymongo.write_concern import WriteConcern
def get_last_version(self, filename: Optional[str]=None, session: Optional[ClientSession]=None, **kwargs: Any) -> GridOut:
    """Get the most recent version of a file in GridFS by ``"filename"``
        or metadata fields.

        Equivalent to calling :meth:`get_version` with the default
        `version` (``-1``).

        :Parameters:
          - `filename`: ``"filename"`` of the file to get, or `None`
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`
          - `**kwargs` (optional): find files by custom metadata.

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
    return self.get_version(filename=filename, session=session, **kwargs)