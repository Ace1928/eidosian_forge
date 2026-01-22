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
def update_one(self, filter: Mapping[str, Any], update: Union[Mapping[str, Any], _Pipeline], upsert: bool=False, bypass_document_validation: bool=False, collation: Optional[_CollationIn]=None, array_filters: Optional[Sequence[Mapping[str, Any]]]=None, hint: Optional[_IndexKeyHint]=None, session: Optional[ClientSession]=None, let: Optional[Mapping[str, Any]]=None, comment: Optional[Any]=None) -> UpdateResult:
    """Update a single document matching the filter.

          >>> for doc in db.test.find():
          ...     print(doc)
          ...
          {'x': 1, '_id': 0}
          {'x': 1, '_id': 1}
          {'x': 1, '_id': 2}
          >>> result = db.test.update_one({'x': 1}, {'$inc': {'x': 3}})
          >>> result.matched_count
          1
          >>> result.modified_count
          1
          >>> for doc in db.test.find():
          ...     print(doc)
          ...
          {'x': 4, '_id': 0}
          {'x': 1, '_id': 1}
          {'x': 1, '_id': 2}

        If ``upsert=True`` and no documents match the filter, create a
        new document based on the filter criteria and update modifications.

          >>> result = db.test.update_one({'x': -10}, {'$inc': {'x': 3}}, upsert=True)
          >>> result.matched_count
          0
          >>> result.modified_count
          0
          >>> result.upserted_id
          ObjectId('626a678eeaa80587d4bb3fb7')
          >>> db.test.find_one(result.upserted_id)
          {'_id': ObjectId('626a678eeaa80587d4bb3fb7'), 'x': -7}

        :Parameters:
          - `filter`: A query that matches the document to update.
          - `update`: The modifications to apply.
          - `upsert` (optional): If ``True``, perform an insert if no documents
            match the filter.
          - `bypass_document_validation`: (optional) If ``True``, allows the
            write to opt-out of document level validation. Default is
            ``False``.
          - `collation` (optional): An instance of
            :class:`~pymongo.collation.Collation`.
          - `array_filters` (optional): A list of filters specifying which
            array elements an update should apply.
          - `hint` (optional): An index to use to support the query
            predicate specified either by its string name, or in the same
            format as passed to
            :meth:`~pymongo.collection.Collection.create_index` (e.g.
            ``[('field', ASCENDING)]``). This option is only supported on
            MongoDB 4.2 and above.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `let` (optional): Map of parameter names and values. Values must be
            constant or closed expressions that do not reference document
            fields. Parameters can then be accessed as variables in an
            aggregate expression context (e.g. "$$var").
          - `comment` (optional): A user-provided comment to attach to this
            command.

        :Returns:
          - An instance of :class:`~pymongo.results.UpdateResult`.

        .. versionchanged:: 4.1
           Added ``let`` parameter.
           Added ``comment`` parameter.
        .. versionchanged:: 3.11
           Added ``hint`` parameter.
        .. versionchanged:: 3.9
           Added the ability to accept a pipeline as the ``update``.
        .. versionchanged:: 3.6
           Added the ``array_filters`` and ``session`` parameters.
        .. versionchanged:: 3.4
          Added the ``collation`` option.
        .. versionchanged:: 3.2
          Added ``bypass_document_validation`` support.

        .. versionadded:: 3.0
        """
    common.validate_is_mapping('filter', filter)
    common.validate_ok_for_update(update)
    common.validate_list_or_none('array_filters', array_filters)
    write_concern = self._write_concern_for(session)
    return UpdateResult(self._update_retryable(filter, update, upsert, write_concern=write_concern, bypass_doc_val=bypass_document_validation, collation=collation, array_filters=array_filters, hint=hint, session=session, let=let, comment=comment), write_concern.acknowledged)