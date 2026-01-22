from pprint import pformat
from six import iteritems
import re
@tcp_socket.setter
def tcp_socket(self, tcp_socket):
    """
        Sets the tcp_socket of this V1Handler.
        TCPSocket specifies an action involving a TCP port. TCP hooks not yet
        supported

        :param tcp_socket: The tcp_socket of this V1Handler.
        :type: V1TCPSocketAction
        """
    self._tcp_socket = tcp_socket