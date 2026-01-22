from __future__ import annotations
from copy import deepcopy
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.dbref import DBRef
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import _DatabaseAggregationCommand
from pymongo.change_stream import DatabaseChangeStream
from pymongo.collection import Collection
from pymongo.command_cursor import CommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.errors import CollectionInvalid, InvalidName, InvalidOperation
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
@_csot.apply
def cursor_command(self, command: Union[str, MutableMapping[str, Any]], value: Any=1, read_preference: Optional[_ServerMode]=None, codec_options: Optional[bson.codec_options.CodecOptions[_CodecDocumentType]]=None, session: Optional[ClientSession]=None, comment: Optional[Any]=None, max_await_time_ms: Optional[int]=None, **kwargs: Any) -> CommandCursor[_DocumentType]:
    """Issue a MongoDB command and parse the response as a cursor.

        If the response from the server does not include a cursor field, an error will be thrown.

        Otherwise, behaves identically to issuing a normal MongoDB command.

        :Parameters:
          - `command`: document representing the command to be issued,
            or the name of the command (for simple commands only).

            .. note:: the order of keys in the `command` document is
               significant (the "verb" must come first), so commands
               which require multiple keys (e.g. `findandmodify`)
               should use an instance of :class:`~bson.son.SON` or
               a string and kwargs instead of a Python `dict`.

          - `value` (optional): value to use for the command verb when
            `command` is passed as a string
          - `read_preference` (optional): The read preference for this
            operation. See :mod:`~pymongo.read_preferences` for options.
            If the provided `session` is in a transaction, defaults to the
            read preference configured for the transaction.
            Otherwise, defaults to
            :attr:`~pymongo.read_preferences.ReadPreference.PRIMARY`.
          - `codec_options`: A :class:`~bson.codec_options.CodecOptions`
            instance.
          - `session` (optional): A
            :class:`~pymongo.client_session.ClientSession`.
          - `comment` (optional): A user-provided comment to attach to future getMores for this
            command.
          - `max_await_time_ms` (optional): The number of ms to wait for more data on future getMores for this command.
          - `**kwargs` (optional): additional keyword arguments will
            be added to the command document before it is sent

        .. note:: :meth:`command` does **not** obey this Database's
           :attr:`read_preference` or :attr:`codec_options`. You must use the
           ``read_preference`` and ``codec_options`` parameters instead.

        .. note:: :meth:`command` does **not** apply any custom TypeDecoders
           when decoding the command response.

        .. note:: If this client has been configured to use MongoDB Stable
           API (see :ref:`versioned-api-ref`), then :meth:`command` will
           automatically add API versioning options to the given command.
           Explicitly adding API versioning options in the command and
           declaring an API version on the client is not supported.

        .. seealso:: The MongoDB documentation on `commands <https://dochub.mongodb.org/core/commands>`_.
        """
    with self.__client._tmp_session(session, close=False) as tmp_session:
        opts = codec_options or DEFAULT_CODEC_OPTIONS
        if read_preference is None:
            read_preference = tmp_session and tmp_session._txn_read_preference() or ReadPreference.PRIMARY
        with self.__client._conn_for_reads(read_preference, tmp_session) as (conn, read_preference):
            response = self._command(conn, command, value, True, None, read_preference, opts, session=tmp_session, **kwargs)
            coll = self.get_collection('$cmd', read_preference=read_preference)
            if response.get('cursor'):
                cmd_cursor = CommandCursor(coll, response['cursor'], conn.address, max_await_time_ms=max_await_time_ms, session=tmp_session, explicit_session=session is not None, comment=comment)
                cmd_cursor._maybe_pin_connection(conn)
                return cmd_cursor
            else:
                raise InvalidOperation('Command does not return a cursor.')