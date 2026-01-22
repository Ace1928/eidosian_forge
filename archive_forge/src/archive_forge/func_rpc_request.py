import uuid
import warnings
from inspect import signature
from typing import Callable, Optional, Tuple, Union, List, Any
import rpcq.messages
def rpc_request(method_name: str, *args, **kwargs) -> rpcq.messages.RPCRequest:
    """
    Create RPC request

    :param method_name: Method name
    :param args: Positional arguments
    :param kwargs: Keyword arguments
    :return: JSON RPC formatted dict
    """
    if args:
        kwargs['*args'] = args
    return rpcq.messages.RPCRequest(jsonrpc='2.0', id=str(uuid.uuid4()), method=method_name, params=kwargs)