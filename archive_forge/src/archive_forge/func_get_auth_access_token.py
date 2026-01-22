import os
import subprocess
import six
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
def get_auth_access_token(account=None):
    """Load user access token with the ``gcloud auth print-access-token`` command.

    Args:
        account (Optional[str]): Account to get the access token for. If not
            specified, the current active account will be used.

    Returns:
        str: The user access token.

    Raises:
        google.auth.exceptions.UserAccessTokenError: if failed to get access
            token from gcloud.
    """
    if os.name == 'nt':
        command = _CLOUD_SDK_WINDOWS_COMMAND
    else:
        command = _CLOUD_SDK_POSIX_COMMAND
    try:
        if account:
            command = (command,) + _CLOUD_SDK_USER_ACCESS_TOKEN_COMMAND + ('--account=' + account,)
        else:
            command = (command,) + _CLOUD_SDK_USER_ACCESS_TOKEN_COMMAND
        access_token = subprocess.check_output(command, stderr=subprocess.STDOUT)
        return access_token.decode('utf-8').strip()
    except (subprocess.CalledProcessError, OSError, IOError) as caught_exc:
        new_exc = exceptions.UserAccessTokenError('Failed to obtain access token', caught_exc)
        six.raise_from(new_exc, caught_exc)