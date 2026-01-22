from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Optional, Union
from bson import _decode_all_selective
from pymongo.errors import NotPrimaryError, OperationFailure
from pymongo.helpers import _check_command_response, _handle_reauth
from pymongo.message import _convert_exception, _GetMore, _OpMsg, _Query
from pymongo.response import PinnedResponse, Response
@_handle_reauth
def run_operation(self, conn: Connection, operation: Union[_Query, _GetMore], read_preference: _ServerMode, listeners: Optional[_EventListeners], unpack_res: Callable[..., list[_DocumentOut]]) -> Response:
    """Run a _Query or _GetMore operation and return a Response object.

        This method is used only to run _Query/_GetMore operations from
        cursors.
        Can raise ConnectionFailure, OperationFailure, etc.

        :Parameters:
          - `conn`: A Connection instance.
          - `operation`: A _Query or _GetMore object.
          - `read_preference`: The read preference to use.
          - `listeners`: Instance of _EventListeners or None.
          - `unpack_res`: A callable that decodes the wire protocol response.
        """
    duration = None
    assert listeners is not None
    publish = listeners.enabled_for_commands
    if publish:
        start = datetime.now()
    use_cmd = operation.use_command(conn)
    more_to_come = operation.conn_mgr and operation.conn_mgr.more_to_come
    if more_to_come:
        request_id = 0
    else:
        message = operation.get_message(read_preference, conn, use_cmd)
        request_id, data, max_doc_size = self._split_message(message)
    if publish:
        cmd, dbn = operation.as_command(conn)
        if '$db' not in cmd:
            cmd['$db'] = dbn
        assert listeners is not None
        listeners.publish_command_start(cmd, dbn, request_id, conn.address, service_id=conn.service_id)
        start = datetime.now()
    try:
        if more_to_come:
            reply = conn.receive_message(None)
        else:
            conn.send_message(data, max_doc_size)
            reply = conn.receive_message(request_id)
        if use_cmd:
            user_fields = _CURSOR_DOC_FIELDS
            legacy_response = False
        else:
            user_fields = None
            legacy_response = True
        docs = unpack_res(reply, operation.cursor_id, operation.codec_options, legacy_response=legacy_response, user_fields=user_fields)
        if use_cmd:
            first = docs[0]
            operation.client._process_response(first, operation.session)
            _check_command_response(first, conn.max_wire_version)
    except Exception as exc:
        if publish:
            duration = datetime.now() - start
            if isinstance(exc, (NotPrimaryError, OperationFailure)):
                failure: _DocumentOut = exc.details
            else:
                failure = _convert_exception(exc)
            assert listeners is not None
            listeners.publish_command_failure(duration, failure, operation.name, request_id, conn.address, service_id=conn.service_id, database_name=dbn)
        raise
    if publish:
        duration = datetime.now() - start
        if use_cmd:
            res: _DocumentOut = docs[0]
        elif operation.name == 'explain':
            res = docs[0] if docs else {}
        else:
            res = {'cursor': {'id': reply.cursor_id, 'ns': operation.namespace()}, 'ok': 1}
            if operation.name == 'find':
                res['cursor']['firstBatch'] = docs
            else:
                res['cursor']['nextBatch'] = docs
        assert listeners is not None
        listeners.publish_command_success(duration, res, operation.name, request_id, conn.address, service_id=conn.service_id, database_name=dbn)
    client = operation.client
    if client and client._encrypter:
        if use_cmd:
            decrypted = client._encrypter.decrypt(reply.raw_command_response())
            docs = _decode_all_selective(decrypted, operation.codec_options, user_fields)
    response: Response
    if client._should_pin_cursor(operation.session) or operation.exhaust:
        conn.pin_cursor()
        if isinstance(reply, _OpMsg):
            more_to_come = reply.more_to_come
        else:
            more_to_come = bool(operation.exhaust and reply.cursor_id)
        if operation.conn_mgr:
            operation.conn_mgr.update_exhaust(more_to_come)
        response = PinnedResponse(data=reply, address=self._description.address, conn=conn, duration=duration, request_id=request_id, from_command=use_cmd, docs=docs, more_to_come=more_to_come)
    else:
        response = Response(data=reply, address=self._description.address, duration=duration, request_id=request_id, from_command=use_cmd, docs=docs)
    return response