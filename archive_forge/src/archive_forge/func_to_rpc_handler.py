import pickle
from abc import ABC, abstractmethod
from types import LambdaType
from typing import Any, Callable, Dict
from uuid import uuid4
from triad import ParamDict, SerializableRLock, assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path, to_type
def to_rpc_handler(obj: Any) -> RPCHandler:
    """Convert object to :class:`~.RPCHandler`. If the object is already
    ``RPCHandler``, then the original instance will be returned.
    If the object is ``None`` then :class:`~.EmptyRPCHandler` will be returned.
    If the object is a python function then :class:`~.RPCFunc` will be returned.

    :param obj: |RPCHandlerLikeObject|
    :return: the RPC handler
    """
    if obj is None:
        return EmptyRPCHandler()
    if isinstance(obj, RPCHandler):
        return obj
    if callable(obj):
        return RPCFunc(obj)
    raise ValueError(obj)