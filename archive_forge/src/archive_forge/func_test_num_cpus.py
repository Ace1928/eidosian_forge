import json
import pytest
from IPython.utils import sysinfo
def test_num_cpus():
    with pytest.deprecated_call():
        sysinfo.num_cpus()