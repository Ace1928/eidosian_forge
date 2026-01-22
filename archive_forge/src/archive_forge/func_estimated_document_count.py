from __future__ import annotations
from collections import abc
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import ASCENDING, _csot, common, helpers, message
from pymongo.aggregation import (
from pymongo.bulk import _Bulk
from pymongo.change_stream import CollectionChangeStream
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor, RawBatchCommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.cursor import Cursor, RawBatchCursor
from pymongo.errors import (
from pymongo.helpers import _check_write_command_response
from pymongo.message import _UNICODE_REPLACE_CODEC_OPTIONS
from pymongo.operations import (
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.results import (
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
from pymongo.write_concern import WriteConcern
def estimated_document_count(self, comment: Optional[Any]=None, **kwargs: Any) -> int:
    """Get an estimate of the number of documents in this collection using
        collection metadata.

        The :meth:`estimated_document_count` method is **not** supported in a
        transaction.

        All optional parameters should be passed as keyword arguments
        to this method. Valid options include:

          - `maxTimeMS` (int): The maximum amount of time to allow this
            operation to run, in milliseconds.

        :Parameters:
          - `comment` (optional): A user-provided comment to attach to this
            command.
          - `**kwargs` (optional): See list of options above.

        .. versionchanged:: 4.2
           This method now always uses the `count`_ command. Due to an oversight in versions
           5.0.0-5.0.8 of MongoDB, the count command was not included in V1 of the
           :ref:`versioned-api-ref`. Users of the Stable API with estimated_document_count are
           recommended to upgrade their server version to 5.0.9+ or set
           :attr:`pymongo.server_api.ServerApi.strict` to ``False`` to avoid encountering errors.

        .. versionadded:: 3.7
        .. _count: https://mongodb.com/docs/manual/reference/command/count/
        """
    if 'session' in kwargs:
        raise ConfigurationError('estimated_document_count does not support sessions')
    if comment is not None:
        kwargs['comment'] = comment

    def _cmd(session: Optional[ClientSession], _server: Server, conn: Connection, read_preference: Optional[_ServerMode]) -> int:
        cmd: SON[str, Any] = SON([('count', self.__name)])
        cmd.update(kwargs)
        return self._count_cmd(session, conn, read_preference, cmd, collation=None)
    return self._retryable_non_cursor_read(_cmd, None)