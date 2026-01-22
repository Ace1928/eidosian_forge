from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def mock_session():
    auth = identity.Password(auth_url='http://localhost/identity/v3', username='username', password='password', project_name='project_name', default_domain_id='default')
    sess = session.Session(auth=auth)
    return sess