import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def test_wrong_usage_orc_writer(tempdir):
    from pyarrow import orc
    path = str(tempdir / 'test.orc')
    with orc.ORCWriter(path) as writer:
        with pytest.raises(AttributeError):
            writer.test()