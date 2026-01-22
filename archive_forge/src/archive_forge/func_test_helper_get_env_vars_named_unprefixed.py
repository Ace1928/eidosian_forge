from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import eq_, ieq, get_env_vars_named
def test_helper_get_env_vars_named_unprefixed(setup_helpers_get_env_vars_named):
    expected = _testcase_get_env_vars_named(3)
    found = get_env_vars_named(GET_ENV_PREFIX, remove_prefix=True)
    ieq(found, expected)