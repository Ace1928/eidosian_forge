import os.path
import pytest
from tempfile import TemporaryDirectory
def test_logstart_unicode():
    with TemporaryDirectory() as tdir:
        logfname = os.path.join(tdir, 'test_unicode.log')
        _ip.run_cell("'abc€'")
        try:
            _ip.magic('logstart -to %s' % logfname)
            _ip.run_cell("'abc€'")
        finally:
            _ip.logger.logstop()