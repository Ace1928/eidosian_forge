import copy
import datetime
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import jwt
from google.oauth2 import _client
def with_always_use_jwt_access(self, always_use_jwt_access):
    """Create a copy of these credentials with the specified always_use_jwt_access value.

        Args:
            always_use_jwt_access (bool): Whether always use self signed JWT or not.

        Returns:
            google.auth.service_account.Credentials: A new credentials
                instance.
        """
    return self.__class__(self._signer, service_account_email=self._service_account_email, scopes=self._scopes, default_scopes=self._default_scopes, token_uri=self._token_uri, subject=self._subject, project_id=self._project_id, quota_project_id=self._quota_project_id, additional_claims=self._additional_claims.copy(), always_use_jwt_access=always_use_jwt_access)