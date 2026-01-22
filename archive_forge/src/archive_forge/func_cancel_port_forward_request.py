import threading
from paramiko import util
from paramiko.common import (
def cancel_port_forward_request(self, address, port):
    """
        The client would like to cancel a previous port-forwarding request.
        If the given address and port is being forwarded across this ssh
        connection, the port should be closed.

        :param str address: the forwarded address
        :param int port: the forwarded port
        """
    pass