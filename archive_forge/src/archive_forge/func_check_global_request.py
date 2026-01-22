import threading
from paramiko import util
from paramiko.common import (
def check_global_request(self, kind, msg):
    """
        Handle a global request of the given ``kind``.  This method is called
        in server mode and client mode, whenever the remote host makes a global
        request.  If there are any arguments to the request, they will be in
        ``msg``.

        There aren't any useful global requests defined, aside from port
        forwarding, so usually this type of request is an extension to the
        protocol.

        If the request was successful and you would like to return contextual
        data to the remote host, return a tuple.  Items in the tuple will be
        sent back with the successful result.  (Note that the items in the
        tuple can only be strings, ints, or bools.)

        The default implementation always returns ``False``, indicating that it
        does not support any global requests.

        .. note:: Port forwarding requests are handled separately, in
            `check_port_forward_request`.

        :param str kind: the kind of global request being made.
        :param .Message msg: any extra arguments to the request.
        :return:
            ``True`` or a `tuple` of data if the request was granted; ``False``
            otherwise.
        """
    return False