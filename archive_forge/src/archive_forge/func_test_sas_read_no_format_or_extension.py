from io import StringIO
import pytest
from pandas import read_sas
import pandas._testing as tm
def test_sas_read_no_format_or_extension(self):
    msg = 'unable to infer format of SAS file.+'
    with tm.ensure_clean('test_file_no_extension') as path:
        with pytest.raises(ValueError, match=msg):
            read_sas(path)