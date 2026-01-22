from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, Type, Union
from bson import CodecOptions, _bson_to_dict
from bson.raw_bson import RawBSONDocument
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import (
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor
from pymongo.errors import (
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
@_csot.apply
def try_next(self) -> Optional[_DocumentType]:
    """Advance the cursor without blocking indefinitely.

        This method returns the next change document without waiting
        indefinitely for the next change. For example::

            with db.collection.watch() as stream:
                while stream.alive:
                    change = stream.try_next()
                    # Note that the ChangeStream's resume token may be updated
                    # even when no changes are returned.
                    print("Current resume token: %r" % (stream.resume_token,))
                    if change is not None:
                        print("Change document: %r" % (change,))
                        continue
                    # We end up here when there are no recent changes.
                    # Sleep for a while before trying again to avoid flooding
                    # the server with getMore requests when no changes are
                    # available.
                    time.sleep(10)

        If no change document is cached locally then this method runs a single
        getMore command. If the getMore yields any documents, the next
        document is returned, otherwise, if the getMore returns no documents
        (because there have been no changes) then ``None`` is returned.

        :Returns:
          The next change document or ``None`` when no document is available
          after running a single getMore or when the cursor is closed.

        .. versionadded:: 3.8
        """
    if not self._closed and (not self._cursor.alive):
        self._resume()
    try:
        try:
            change = self._cursor._try_next(True)
        except PyMongoError as exc:
            if not _resumable(exc):
                raise
            self._resume()
            change = self._cursor._try_next(False)
    except PyMongoError as exc:
        if not _resumable(exc) and (not exc.timeout):
            self.close()
        raise
    except Exception:
        self.close()
        raise
    if not self._cursor.alive:
        self._closed = True
    if change is None:
        if self._cursor._post_batch_resume_token is not None:
            self._resume_token = self._cursor._post_batch_resume_token
            self._start_at_operation_time = None
        return change
    try:
        resume_token = change['_id']
    except KeyError:
        self.close()
        raise InvalidOperation('Cannot provide resume functionality when the resume token is missing.') from None
    if not self._cursor._has_next() and self._cursor._post_batch_resume_token:
        resume_token = self._cursor._post_batch_resume_token
    self._uses_start_after = False
    self._uses_resume_after = True
    self._resume_token = resume_token
    self._start_at_operation_time = None
    if self._decode_custom:
        return _bson_to_dict(change.raw, self._orig_codec_options)
    return change