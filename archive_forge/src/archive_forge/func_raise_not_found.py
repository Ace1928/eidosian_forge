from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
import simplejson as json
from osc_lib import exceptions
from osc_lib.i18n import _
def raise_not_found():
    msg = _('%s not found') % value
    raise exceptions.NotFound(404, msg)