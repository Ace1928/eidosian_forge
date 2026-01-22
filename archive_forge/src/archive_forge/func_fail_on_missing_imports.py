from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.six import PY2
def fail_on_missing_imports():
    if REQUESTS_IMPORT_ERROR is not None:
        from .errors import MissingRequirementException
        raise MissingRequirementException('You have to install requests', 'requests', REQUESTS_IMPORT_ERROR)
    if URLLIB3_IMPORT_ERROR is not None:
        from .errors import MissingRequirementException
        raise MissingRequirementException('You have to install urllib3', 'urllib3', URLLIB3_IMPORT_ERROR)
    if BACKPORTS_SSL_MATCH_HOSTNAME_IMPORT_ERROR is not None:
        from .errors import MissingRequirementException
        raise MissingRequirementException('You have to install backports.ssl-match-hostname', 'backports.ssl-match-hostname', BACKPORTS_SSL_MATCH_HOSTNAME_IMPORT_ERROR)