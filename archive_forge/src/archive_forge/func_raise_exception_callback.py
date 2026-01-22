import stevedore
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def raise_exception_callback(manager, entrypoint, exc):
    error = "Cannot load '%(entrypoint)s' entry_point: %(error)s'" % {'entrypoint': entrypoint, 'error': exc}
    errors.append(error)