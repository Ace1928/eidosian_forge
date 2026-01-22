import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
@pytest.mark.parametrize('options', [(None, None), ({'GET'}, None), (None, {'GET'}), ({'GET'}, {'GET'})])
def test_retry_allowed_methods_and_method_whitelist_error(self, options):
    with pytest.raises(ValueError) as e:
        Retry(allowed_methods=options[0], method_whitelist=options[1])
    assert str(e.value) == "Using both 'allowed_methods' and 'method_whitelist' together is not allowed. Instead only use 'allowed_methods'"