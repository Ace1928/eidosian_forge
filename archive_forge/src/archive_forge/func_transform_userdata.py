import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
@staticmethod
def transform_userdata(userdata):
    if hasattr(userdata, 'read'):
        userdata = userdata.read()
    try:
        userdata = userdata.encode('utf-8')
    except AttributeError:
        pass
    return base64.b64encode(userdata).decode('utf-8')