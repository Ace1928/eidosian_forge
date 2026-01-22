import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def get_prop_list_entry_value(self, prop_list_p, prop_list_sz, entry_name, entry_type, entry_syntax):
    prop_entry = self.get_prop_list_entry_p(prop_list_p, prop_list_sz, entry_name)
    if prop_entry['length'] != ctypes.sizeof(entry_type) or prop_entry['syntax'] != entry_syntax:
        raise exceptions.ClusterPropertyListParsingError()
    return entry_type.from_address(prop_entry['val_p'].value).value