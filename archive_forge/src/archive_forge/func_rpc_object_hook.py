import base64
import errno
import json
from multiprocessing import connection
from multiprocessing import managers
import socket
import struct
import weakref
from oslo_rootwrap import wrapper
def rpc_object_hook(obj):
    if '__exception__' in obj:
        type_name = obj.pop('__exception__')
        if type_name not in ('NoFilterMatched', 'FilterMatchNotExecutable'):
            return obj
        exc_type = getattr(wrapper, type_name)
        return exc_type(**obj)
    elif '__bytes__' in obj:
        return base64.b64decode(obj['__bytes__'].encode('ascii'))
    else:
        return obj