import sys
import pytest
from cheroot.cli import (
@pytest.mark.parametrize(('app_name', 'app_method'), ((None, 'application'), ('application', 'application'), ('main', 'main')))
def test_Aplication_resolve(app_name, app_method, wsgi_app):
    """Check the wsgi application name conversion."""
    if app_name is None:
        wsgi_app_spec = 'mypkg.wsgi'
    else:
        wsgi_app_spec = 'mypkg.wsgi:{app_name}'.format(**locals())
    expected_app = getattr(wsgi_app, app_method)
    assert Application.resolve(wsgi_app_spec).wsgi_app == expected_app