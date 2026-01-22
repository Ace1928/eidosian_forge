import json
import os
import re
from pandas.util._print_versions import (
import pandas as pd
def test_show_versions_console_json(capsys):
    pd.show_versions(as_json=True)
    stdout = capsys.readouterr().out
    result = json.loads(stdout)
    expected = {'system': _get_sys_info(), 'dependencies': _get_dependency_info()}
    assert result == expected