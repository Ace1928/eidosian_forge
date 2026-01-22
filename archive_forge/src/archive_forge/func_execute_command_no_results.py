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
def execute_command_no_results(self, conn: Connection, generator: Iterator[Any], write_concern: WriteConcern) -> None:
    """Execute write commands with OP_MSG and w=0 WriteConcern, ordered."""
    full_result = {'writeErrors': [], 'writeConcernErrors': [], 'nInserted': 0, 'nUpserted': 0, 'nMatched': 0, 'nModified': 0, 'nRemoved': 0, 'upserted': []}
    initial_write_concern = WriteConcern()
    op_id = _randint()
    try:
        self._execute_command(generator, initial_write_concern, None, conn, op_id, False, full_result, write_concern)
    except OperationFailure:
        pass