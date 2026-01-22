import io
import json
import logging
import os
import warnings
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client
def load_credentials_from_dict(info, scopes=None, default_scopes=None, quota_project_id=None, request=None):
    """Loads Google credentials from a dict.

    The credentials file must be a service account key, stored authorized
    user credentials, external account credentials, or impersonated service
    account credentials.

    Args:
        info (Dict[str, Any]): A dict object containing the credentials
        scopes (Optional[Sequence[str]]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary
        default_scopes (Optional[Sequence[str]]): Default scopes passed by a
            Google client library. Use 'scopes' for user-defined scopes.
        quota_project_id (Optional[str]):  The project ID used for
            quota and billing.
        request (Optional[google.auth.transport.Request]): An object used to make
            HTTP requests. This is used to determine the associated project ID
            for a workload identity pool resource (external account credentials).
            If not specified, then it will use a
            google.auth.transport.requests.Request client to make requests.

    Returns:
        Tuple[google.auth.credentials.Credentials, Optional[str]]: Loaded
            credentials and the project ID. Authorized user credentials do not
            have the project ID information. External account credentials project
            IDs may not always be determined.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: if the file is in the
            wrong format or is missing.
    """
    if not isinstance(info, dict):
        raise exceptions.DefaultCredentialsError('info object was of type {} but dict type was expected.'.format(type(info)))
    return _load_credentials_from_info('dict object', info, scopes, default_scopes, quota_project_id, request)