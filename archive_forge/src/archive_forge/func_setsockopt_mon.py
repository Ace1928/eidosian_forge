import zmq
from zmq.devices.basedevice import Device, ProcessDevice, ThreadDevice
def setsockopt_mon(self, opt, value):
    """Enqueue setsockopt(opt, value) for mon_socket

        See zmq.Socket.setsockopt for details.
        """
    self._mon_sockopts.append((opt, value))