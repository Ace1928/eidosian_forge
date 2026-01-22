import errno as errno_mod
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import ZMQError, _check_rc, _check_version
from ._cffi import ffi
from ._cffi import lib as C
from .message import Frame
from .utils import _retry_sys_call
def new_pointer_from_opt(option, length=0):
    opt_type = getattr(option, '_opt_type', _OptType.int)
    if opt_type == _OptType.int64 or (ZMQ_FD_64BIT and opt_type == _OptType.fd):
        return new_int64_pointer()
    elif opt_type == _OptType.bytes:
        return new_binary_data(length)
    else:
        return new_int_pointer()