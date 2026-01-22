import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def new_sp():
    return {'id': uuid.uuid4().hex, 'sp_url': uuid.uuid4().hex, 'auth_url': uuid.uuid4().hex}