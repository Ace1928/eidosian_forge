import collections
from functools import partial
import string
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version
@pytest.mark.single_cpu
def test_bz2_missing_import():
    code = "\n        import sys\n        sys.modules['bz2'] = None\n        import pytest\n        import pandas as pd\n        from pandas.compat import get_bz2_file\n        msg = 'bz2 module not available.'\n        with pytest.raises(RuntimeError, match=msg):\n            get_bz2_file()\n    "
    code = textwrap.dedent(code)
    call = [sys.executable, '-c', code]
    subprocess.check_output(call)