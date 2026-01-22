import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
def ssh_check_mic(self, mic_token, session_id, username=None):
    """
        Verify the MIC token for a SSH2 message.

        :param str mic_token: The MIC token received from the client
        :param str session_id: The SSH session ID
        :param str username: The name of the user who attempts to login
        :return: None if the MIC check was successful
        :raises: ``sspi.error`` -- if the MIC check failed
        """
    self._session_id = session_id
    self._username = username
    if username is not None:
        mic_field = self._ssh_build_mic(self._session_id, self._username, self._service, self._auth_method)
        self._gss_srv_ctxt.verify(mic_field, mic_token)
    else:
        self._gss_ctxt.verify(self._session_id, mic_token)