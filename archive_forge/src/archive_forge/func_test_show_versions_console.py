import json
import os
import re
from pandas.util._print_versions import (
import pandas as pd
def test_show_versions_console(capsys):
    pd.show_versions(as_json=False)
    result = capsys.readouterr().out
    assert 'INSTALLED VERSIONS' in result
    assert re.search('commit\\s*:\\s[0-9a-f]{40}\\n', result)
    assert re.search('numpy\\s*:\\s[0-9]+\\..*\\n', result)
    assert re.search('pyarrow\\s*:\\s([0-9]+.*|None)\\n', result)