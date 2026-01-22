from __future__ import absolute_import, unicode_literals
import datetime
import os
from oauthlib.common import unicode_type, urldecode
def is_secure_transport(uri):
    """Check if the uri is over ssl."""
    if os.environ.get('OAUTHLIB_INSECURE_TRANSPORT'):
        return True
    return uri.lower().startswith('https://')