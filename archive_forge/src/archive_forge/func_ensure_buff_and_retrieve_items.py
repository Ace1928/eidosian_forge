import ctypes
import functools
import inspect
import socket
import time
from oslo_log import log as logging
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def ensure_buff_and_retrieve_items(struct_type, func_requests_buff_sz=True, parse_output=True):

    def wrapper(f):

        @functools.wraps(f)
        def inner(*args, **kwargs):
            call_args = inspect.getcallargs(f, *args, **kwargs)
            call_args['element_count'] = ctypes.c_ulong(0)
            call_args['buff'] = _get_buff(0, struct_type)
            call_args['buff_size'] = ctypes.c_ulong(0)
            while True:
                try:
                    ret_val = f(**call_args)
                    if parse_output:
                        return _get_items_from_buff(call_args['buff'], struct_type, call_args['element_count'].value)
                    else:
                        return ret_val
                except exceptions.Win32Exception as ex:
                    if ex.error_code & 65535 == w_const.ERROR_INSUFFICIENT_BUFFER:
                        if func_requests_buff_sz:
                            buff_size = call_args['buff_size'].value
                        else:
                            buff_size = ctypes.sizeof(struct_type) * call_args['element_count'].value
                        call_args['buff'] = _get_buff(buff_size, struct_type)
                    else:
                        raise
        return inner
    return wrapper