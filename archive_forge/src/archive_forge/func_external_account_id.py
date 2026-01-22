import json
import os
import subprocess
import sys
import time
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
@property
def external_account_id(self):
    """Returns the external account identifier.

        When service account impersonation is used the identifier is the service
        account email.

        Without service account impersonation, this returns None, unless it is
        being used by the Google Cloud CLI which populates this field.
        """
    return self.service_account_email or self._tokeninfo_username