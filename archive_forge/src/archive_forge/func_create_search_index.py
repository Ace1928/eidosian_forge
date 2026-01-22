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
def create_search_index(self, model: Union[Mapping[str, Any], SearchIndexModel], session: Optional[ClientSession]=None, comment: Any=None, **kwargs: Any) -> str:
    """Create a single search index for the current collection.

        :Parameters:
          - `model`: The model for the new search index.
            It can be given as a :class:`~pymongo.operations.SearchIndexModel`
            instance or a dictionary with a model "definition"  and optional
            "name".
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `comment` (optional): A user-provided comment to attach to this
            command.
          - `**kwargs` (optional): optional arguments to the createSearchIndexes
            command (like maxTimeMS) can be passed as keyword arguments.

        :Returns:
          The name of the new search index.

        .. note:: requires a MongoDB server version 7.0+ Atlas cluster.

        .. versionadded:: 4.5
        """
    if not isinstance(model, SearchIndexModel):
        model = SearchIndexModel(model['definition'], model.get('name'))
    return self.create_search_indexes([model], session, comment, **kwargs)[0]