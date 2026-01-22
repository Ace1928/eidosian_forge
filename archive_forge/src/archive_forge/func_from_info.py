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
@classmethod
def from_info(cls, info, **kwargs):
    """Creates a Credentials instance from parsed external account info.

        Args:
            info (Mapping[str, str]): The external account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.identity_pool.Credentials: The constructed
                credentials.

        Raises:
            InvalidValue: For invalid parameters.
        """
    return cls(audience=info.get('audience'), subject_token_type=info.get('subject_token_type'), token_url=info.get('token_url'), token_info_url=info.get('token_info_url'), service_account_impersonation_url=info.get('service_account_impersonation_url'), service_account_impersonation_options=info.get('service_account_impersonation') or {}, client_id=info.get('client_id'), client_secret=info.get('client_secret'), credential_source=info.get('credential_source'), quota_project_id=info.get('quota_project_id'), workforce_pool_user_project=info.get('workforce_pool_user_project'), universe_domain=info.get('universe_domain', credentials.DEFAULT_UNIVERSE_DOMAIN), **kwargs)