import os
import sys
from .. import bedding, osutils, tests
def mock_special_folder_path(self, csidl):
    if csidl == win32utils.CSIDL_APPDATA:
        return self.appdata
    elif csidl == win32utils.CSIDL_PERSONAL:
        return self.test_dir
    return None