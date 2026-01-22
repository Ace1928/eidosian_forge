from __future__ import absolute_import, unicode_literals
import logging
def validate_user(self, username, password, client, request, *args, **kwargs):
    """Ensure the username and password is valid.

        OBS! The validation should also set the user attribute of the request
        to a valid resource owner, i.e. request.user = username or similar. If
        not set you will be unable to associate a token with a user in the
        persistance method used (commonly, save_bearer_token).

        :param username: Unicode username.
        :param password: Unicode password.
        :param client: Client object set by you, see ``.authenticate_client``.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Resource Owner Password Credentials Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')