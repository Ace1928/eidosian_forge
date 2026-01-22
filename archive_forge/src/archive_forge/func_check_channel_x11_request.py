import threading
from paramiko import util
from paramiko.common import (
def check_channel_x11_request(self, channel, single_connection, auth_protocol, auth_cookie, screen_number):
    """
        Determine if the client will be provided with an X11 session.  If this
        method returns ``True``, X11 applications should be routed through new
        SSH channels, using `.Transport.open_x11_channel`.

        The default implementation always returns ``False``.

        :param .Channel channel: the `.Channel` the X11 request arrived on
        :param bool single_connection:
            ``True`` if only a single X11 channel should be opened, else
            ``False``.
        :param str auth_protocol: the protocol used for X11 authentication
        :param str auth_cookie: the cookie used to authenticate to X11
        :param int screen_number: the number of the X11 screen to connect to
        :return: ``True`` if the X11 session was opened; ``False`` if not
        """
    return False