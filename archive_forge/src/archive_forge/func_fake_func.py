import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
@iscsi_utils.ensure_buff_and_retrieve_items(**decorator_args)
def fake_func(inst, buff=None, buff_size=None, element_count=None, *args, **kwargs):
    raised_exc = None
    try:
        self.assertIsInstance(buff_size, ctypes.c_ulong)
        self.assertIsInstance(element_count, ctypes.c_ulong)
        func_side_effect(*args, buff=buff, buff_size_val=buff_size.value, element_count_val=element_count.value, **kwargs)
    except Exception as ex:
        raised_exc = ex
    if returned_element_count:
        element_count.value = returned_element_count
    if required_buff_sz:
        buff_size.value = required_buff_sz
    if raised_exc:
        raise raised_exc
    return mock.sentinel.ret_val