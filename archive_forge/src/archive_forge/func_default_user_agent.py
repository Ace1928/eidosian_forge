import os
import ssl
import sys
from ... import config
from ... import version_string as breezy_version
def default_user_agent():
    return 'Breezy/%s' % breezy_version