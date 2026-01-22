import json
import pytest
from IPython.utils import sysinfo
def test_json_getsysinfo():
    """
    test that it is easily jsonable and don't return bytes somewhere.
    """
    json.dumps(sysinfo.get_sys_info())