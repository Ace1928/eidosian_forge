from collections import deque
import select
import msgpack
def receive_messages(self, all=False):
    """Try to receive some messages.
        Received messages are put on the internal queues.
        They can be retrieved using get_xxx() methods.
        Returns True if there's something queued for get_xxx() methods.
        """
    while all or self._incoming == 0:
        try:
            packet = self._sock.recv(4096)
        except IOError:
            packet = None
        if not packet:
            if packet is not None:
                self._closed_by_peer = True
            break
        self._encoder.get_and_dispatch_messages(packet, self._table)
    return self._incoming > 0