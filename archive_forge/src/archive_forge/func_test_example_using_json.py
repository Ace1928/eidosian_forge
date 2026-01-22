import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
@pytest.mark.pandas
@pytest.mark.parametrize('filename', ['TestOrcFile.test1.orc', 'TestOrcFile.testDate1900.orc', 'decimal.orc'])
def test_example_using_json(filename, datadir):
    """
    Check a ORC file example against the equivalent JSON file, as given
    in the Apache ORC repository (the JSON file has one JSON object per
    line, corresponding to one row in the ORC file).
    """
    path = datadir / filename
    table = pd.read_json(str(path.with_suffix('.jsn.gz')), lines=True)
    check_example_file(path, table, need_fix=True)