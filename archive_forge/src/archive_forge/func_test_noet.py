import pytest
from ..config import ET_ROOT
from ..client import _etrequest, get_project, check_available_version
def test_noet():
    import os
    old_var = None
    if 'NO_ET' in os.environ:
        old_var = (True, os.environ['NO_ET'])
    os.environ['NO_ET'] = '1'
    repo = 'github/hub'
    res = get_project(repo)
    assert res is None
    if old_var is None:
        del os.environ['NO_ET']
    else:
        os.environ['NO_ET'] = old_var[1]