import re
import pytest
from pandas.core.indexes.frozen import FrozenList
def check_mutable_error(self, *args, **kwargs):
    mutable_regex = re.compile('does not support mutable operations')
    msg = "'(_s)?re.(SRE_)?Pattern' object is not callable"
    with pytest.raises(TypeError, match=msg):
        mutable_regex(*args, **kwargs)