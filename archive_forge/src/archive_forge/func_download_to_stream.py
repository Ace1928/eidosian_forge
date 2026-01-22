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
@_csot.apply
def download_to_stream(self, file_id: Any, destination: Any, session: Optional[ClientSession]=None) -> None:
    """Downloads the contents of the stored file specified by file_id and
        writes the contents to `destination`.

        For example::

          my_db = MongoClient().test
          fs = GridFSBucket(my_db)
          # Get _id of file to read
          file_id = fs.upload_from_stream("test_file", "data I want to store!")
          # Get file to write to
          file = open('myfile','wb+')
          fs.download_to_stream(file_id, file)
          file.seek(0)
          contents = file.read()

        Raises :exc:`~gridfs.errors.NoFile` if no file with file_id exists.

        :Parameters:
          - `file_id`: The _id of the file to be downloaded.
          - `destination`: a file-like object implementing :meth:`write`.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
    with self.open_download_stream(file_id, session=session) as gout:
        while True:
            chunk = gout.readchunk()
            if not len(chunk):
                break
            destination.write(chunk)