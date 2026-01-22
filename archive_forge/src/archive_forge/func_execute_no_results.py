from __future__ import annotations
import copy
from collections.abc import MutableMapping
from itertools import islice
from typing import (
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from pymongo import _csot, common
from pymongo.client_session import ClientSession, _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES, _get_wce_doc
from pymongo.message import (
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def execute_no_results(self, conn: Connection, generator: Iterator[Any], write_concern: WriteConcern) -> None:
    """Execute all operations, returning no results (w=0)."""
    if self.uses_collation:
        raise ConfigurationError('Collation is unsupported for unacknowledged writes.')
    if self.uses_array_filters:
        raise ConfigurationError('arrayFilters is unsupported for unacknowledged writes.')
    unack = write_concern and (not write_concern.acknowledged)
    if unack and self.uses_hint_delete and (conn.max_wire_version < 9):
        raise ConfigurationError('Must be connected to MongoDB 4.4+ to use hint on unacknowledged delete commands.')
    if unack and self.uses_hint_update and (conn.max_wire_version < 8):
        raise ConfigurationError('Must be connected to MongoDB 4.2+ to use hint on unacknowledged update commands.')
    if self.bypass_doc_val:
        raise OperationFailure('Cannot set bypass_document_validation with unacknowledged write concern')
    if self.ordered:
        return self.execute_command_no_results(conn, generator, write_concern)
    return self.execute_op_msg_no_results(conn, generator)