import zmq
from zmq.devices.proxydevice import ProcessProxy, Proxy, ThreadProxy
def bind_ctrl_to_random_port(self, addr, *args, **kwargs):
    """Enqueue a random port on the given interface for binding on
        ctrl_socket.

        See zmq.Socket.bind_to_random_port for details.
        """
    port = self._reserve_random_port(addr, *args, **kwargs)
    self.bind_ctrl('%s:%i' % (addr, port))
    return port