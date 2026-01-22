from __future__ import annotations
import contextlib
import enum
import socket
import weakref
from copy import deepcopy
from typing import (
from bson import _dict_to_bson, decode, encode
from bson.binary import STANDARD, UUID_SUBTYPE, Binary
from bson.codec_options import CodecOptions
from bson.errors import BSONError
from bson.raw_bson import DEFAULT_RAW_BSON_OPTIONS, RawBSONDocument, _inflate_bson
from bson.son import SON
from pymongo import _csot
from pymongo.collection import Collection
from pymongo.common import CONNECT_TIMEOUT
from pymongo.cursor import Cursor
from pymongo.daemon import _spawn_daemon
from pymongo.database import Database
from pymongo.encryption_options import AutoEncryptionOpts, RangeOpts
from pymongo.errors import (
from pymongo.mongo_client import MongoClient
from pymongo.network import BLOCKING_IO_ERRORS
from pymongo.operations import UpdateOne
from pymongo.pool import PoolOptions, _configured_socket, _raise_connection_failure
from pymongo.read_concern import ReadConcern
from pymongo.results import BulkWriteResult, DeleteResult
from pymongo.ssl_support import get_ssl_context
from pymongo.typings import _DocumentType, _DocumentTypeArg
from pymongo.uri_parser import parse_host
from pymongo.write_concern import WriteConcern
def mark_command(self, database: str, cmd: bytes) -> bytes:
    """Mark a command for encryption.

        :Parameters:
          - `database`: The database on which to run this command.
          - `cmd`: The BSON command to run.

        :Returns:
          The marked command response from mongocryptd.
        """
    if not self._spawned and (not self.opts._mongocryptd_bypass_spawn):
        self.spawn()
    inflated_cmd = _inflate_bson(cmd, DEFAULT_RAW_BSON_OPTIONS)
    assert self.mongocryptd_client is not None
    try:
        res = self.mongocryptd_client[database].command(inflated_cmd, codec_options=DEFAULT_RAW_BSON_OPTIONS)
    except ServerSelectionTimeoutError:
        if self.opts._mongocryptd_bypass_spawn:
            raise
        self.spawn()
        res = self.mongocryptd_client[database].command(inflated_cmd, codec_options=DEFAULT_RAW_BSON_OPTIONS)
    return res.raw