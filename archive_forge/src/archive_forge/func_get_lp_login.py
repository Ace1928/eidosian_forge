from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
def get_lp_login(_config=None):
    """Return the user's Launchpad username.

    :raises: MismatchedUsername if authentication.conf and breezy.conf
        disagree about username.
    """
    if _config is None:
        _config = GlobalStack()
    username = _config.get('launchpad_username')
    if username is not None:
        auth = AuthenticationConfig()
        auth_username = _get_auth_user(auth)
        if auth_username is None:
            trace.note(gettext('Setting ssh/sftp usernames for launchpad.net.'))
            _set_auth_user(username, auth)
        elif auth_username != username:
            raise MismatchedUsernames()
    return username