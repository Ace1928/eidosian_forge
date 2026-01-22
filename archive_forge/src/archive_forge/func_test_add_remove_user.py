from pyxnat import Interface
from requests.exceptions import ConnectionError
import os.path as op
from functools import wraps
import pytest
@docker_available
def test_add_remove_user():
    x = Interface(config='.xnat.cfg')
    x.select.project('nosetests5').remove_user('admin')
    x.select.project('nosetests5').add_user('admin', 'collaborator')
    assert 'admin' in x.select.project('nosetests5').collaborators()
    x.select.project('nosetests5').remove_user('admin')
    assert 'admin' not in x.select.project('nosetests5').collaborators()
    x.select.project('nosetests5').add_user('admin', 'owner')