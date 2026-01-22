from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import eq_, ieq, get_env_vars_named
@pytest.fixture()
def setup_helpers_get_env_vars_named(monkeypatch):
    varlist = _testcase_get_env_vars_named(3, prefix=GET_ENV_PREFIX)
    for k, v in varlist.items():
        monkeypatch.setenv(k, v)