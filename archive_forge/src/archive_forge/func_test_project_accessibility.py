from pyxnat import Interface
from requests.exceptions import ConnectionError
import os.path as op
from functools import wraps
import pytest
@docker_available
def test_project_accessibility():
    x = Interface(config='.xnat.cfg')
    print(x.select.project('nosetests5').accessibility())
    assert x.select.project('nosetests5').accessibility() in [b'public', b'protected', b'private']
    x.select.project('nosetests5').set_accessibility('private')
    assert x.select.project('nosetests5').accessibility() == b'private'
    x.select.project('nosetests5').set_accessibility('protected')
    assert x.select.project('nosetests5').accessibility() == b'protected'