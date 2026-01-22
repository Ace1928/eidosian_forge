from pyxnat import Interface
from requests.exceptions import ConnectionError
import os.path as op
from functools import wraps
import pytest
@docker_available
def test_user_lastname():
    x = Interface(config='.xnat.cfg')
    assert x.manage.users.lastname('admin') == 'Admin'