import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def set_session_config(self, session_user_agent_name, session_user_agent_version, session_user_agent_extra):
    """
        Set the user agent configuration values that apply at session level.

        :param user_agent_name: The user agent name configured in the
            :py:class:`botocore.session.Session` object. For backwards
            compatibility, this will always be at the beginning of the
            User-Agent string, together with ``user_agent_version``.
        :param user_agent_version: The user agent version configured in the
            :py:class:`botocore.session.Session` object.
        :param user_agent_extra: The user agent "extra" configured in the
            :py:class:`botocore.session.Session` object.
        """
    self._session_user_agent_name = session_user_agent_name
    self._session_user_agent_version = session_user_agent_version
    self._session_user_agent_extra = session_user_agent_extra
    return self