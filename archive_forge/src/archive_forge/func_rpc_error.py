import uuid
import warnings
from inspect import signature
from typing import Callable, Optional, Tuple, Union, List, Any
import rpcq.messages
def rpc_error(id: Union[str, int], error_msg: str, warnings: List[Any]=[]) -> rpcq.messages.RPCError:
    """
    Create RPC error

    :param id: Request ID
    :param error_msg: Error message
    :param warning: List of warnings to attach to the message
    :return: JSON RPC formatted dict
    """
    return rpcq.messages.RPCError(jsonrpc='2.0', id=id, error=error_msg, warnings=[rpc_warning(warning) for warning in warnings])