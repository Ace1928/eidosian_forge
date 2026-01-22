import json
import os
import pytest  # type: ignore
import six
from google.auth import _service_account_info
from google.auth import crypt
def test_from_dict_bad_private_key():
    info = SERVICE_ACCOUNT_INFO.copy()
    info['private_key'] = 'garbage'
    with pytest.raises(ValueError) as excinfo:
        _service_account_info.from_dict(info)
    assert excinfo.match('key')