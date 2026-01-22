import errno as errno_mod
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import ZMQError, _check_rc, _check_version
from ._cffi import ffi
from ._cffi import lib as C
from .message import Frame
from .utils import _retry_sys_call
def initialize_opt_pointer(option, value, length=0):
    opt_type = getattr(option, '_opt_type', _OptType.int)
    if opt_type == _OptType.int64 or (ZMQ_FD_64BIT and opt_type == _OptType.fd):
        return value_int64_pointer(value)
    elif opt_type == _OptType.bytes:
        return value_binary_data(value, length)
    else:
        return value_int_pointer(value)