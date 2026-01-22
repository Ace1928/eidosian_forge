from pyxnat import Interface
from requests.exceptions import ConnectionError
import os.path as op
from functools import wraps
import pytest
def check_and_raise():
    if 'setup_docker_xnat' in func.__name__:
        print('Initializing XNAT.')
        return
    fp = op.abspath('.xnat.cfg')
    print(fp, op.isfile(fp))
    x = Interface(config=op.abspath('.xnat.cfg'))
    try:
        x.head('')
        list(x.select.projects())
        print('Docker instance found.')
    except (ConnectionError, KeyError):
        print('Skipping it.')
        pytest.skip('Docker-based XNAT instance unavailable')