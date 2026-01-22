from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def await_promise(promise_object_id: RemoteObjectId, return_by_value: typing.Optional[bool]=None, generate_preview: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    """
    Add handler to promise with given promise object id.

    :param promise_object_id: Identifier of the promise.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: *(Optional)* Whether preview should be generated for the result.
    :returns: A tuple with the following items:

        0. **result** - Promise result. Will contain rejected value if promise was rejected.
        1. **exceptionDetails** - *(Optional)* Exception details if stack strace is available.
    """
    params: T_JSON_DICT = dict()
    params['promiseObjectId'] = promise_object_id.to_json()
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.awaitPromise', 'params': params}
    json = (yield cmd_dict)
    return (RemoteObject.from_json(json['result']), ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None)