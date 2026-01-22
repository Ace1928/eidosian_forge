import zmq
from zmq.devices.basedevice import Device, ProcessDevice, ThreadDevice
def bind_mon(self, addr):
    """Enqueue ZMQ address for binding on mon_socket.

        See zmq.Socket.bind for details.
        """
    self._mon_binds.append(addr)