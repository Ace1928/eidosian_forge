import os
import ssl
import sys
from ... import config
from ... import version_string as breezy_version
def default_ca_reqs():
    if sys.platform in ('win32', 'darwin'):
        return 'none'
    else:
        return 'required'