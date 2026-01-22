import io
import json
import os
import six
from google.auth import _default
from google.auth import environment_vars
from google.auth import exceptions
def load_credentials_from_file(filename, scopes=None, quota_project_id=None):
    """Loads Google credentials from a file.

    The credentials file must be a service account key or stored authorized
    user credentials.

    Args:
        filename (str): The full path to the credentials file.
        scopes (Optional[Sequence[str]]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary
        quota_project_id (Optional[str]):  The project ID used for
                quota and billing.

    Returns:
        Tuple[google.auth.credentials.Credentials, Optional[str]]: Loaded
            credentials and the project ID. Authorized user credentials do not
            have the project ID information.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: if the file is in the
            wrong format or is missing.
    """
    if not os.path.exists(filename):
        raise exceptions.DefaultCredentialsError('File {} was not found.'.format(filename))
    with io.open(filename, 'r') as file_obj:
        try:
            info = json.load(file_obj)
        except ValueError as caught_exc:
            new_exc = exceptions.DefaultCredentialsError('File {} is not a valid json file.'.format(filename), caught_exc)
            six.raise_from(new_exc, caught_exc)
    credential_type = info.get('type')
    if credential_type == _default._AUTHORIZED_USER_TYPE:
        from google.oauth2 import _credentials_async as credentials
        try:
            credentials = credentials.Credentials.from_authorized_user_info(info, scopes=scopes)
        except ValueError as caught_exc:
            msg = 'Failed to load authorized user credentials from {}'.format(filename)
            new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
            six.raise_from(new_exc, caught_exc)
        if quota_project_id:
            credentials = credentials.with_quota_project(quota_project_id)
        if not credentials.quota_project_id:
            _default._warn_about_problematic_credentials(credentials)
        return (credentials, None)
    elif credential_type == _default._SERVICE_ACCOUNT_TYPE:
        from google.oauth2 import _service_account_async as service_account
        try:
            credentials = service_account.Credentials.from_service_account_info(info, scopes=scopes).with_quota_project(quota_project_id)
        except ValueError as caught_exc:
            msg = 'Failed to load service account credentials from {}'.format(filename)
            new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
            six.raise_from(new_exc, caught_exc)
        return (credentials, info.get('project_id'))
    else:
        raise exceptions.DefaultCredentialsError('The file {file} does not have a valid type. Type is {type}, expected one of {valid_types}.'.format(file=filename, type=credential_type, valid_types=_default._VALID_TYPES))