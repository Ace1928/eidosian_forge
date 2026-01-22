import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
def finish_sendfile(trans, key, ov):
    try:
        return ov.getresult()
    except OSError as exc:
        if exc.winerror in (_overlapped.ERROR_NETNAME_DELETED, _overlapped.ERROR_OPERATION_ABORTED):
            raise ConnectionResetError(*exc.args)
        else:
            raise