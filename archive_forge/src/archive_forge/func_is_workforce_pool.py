from the current environment without the need to copy, save and manage
import abc
import copy
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
@property
def is_workforce_pool(self):
    """Returns whether the credentials represent a workforce pool (True) or
        workload (False) based on the credentials' audience.

        This will also return True for impersonated workforce pool credentials.

        Returns:
            bool: True if the credentials represent a workforce pool. False if they
                represent a workload.
        """
    p = re.compile('//iam\\.googleapis\\.com/locations/[^/]+/workforcePools/')
    return p.match(self._audience or '') is not None