from collections import deque
import select
import msgpack
def process_outgoing(self):
    try:
        sent_bytes = self._sock.send(self._send_buffer)
    except IOError:
        sent_bytes = 0
    del self._send_buffer[:sent_bytes]