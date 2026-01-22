from collections import deque
import select
import msgpack
def get_and_dispatch_messages(self, data, disp_table):
    """dissect messages from a raw stream data.
        disp_table[type] should be a callable for the corresponding
        MessageType.
        """
    self._unpacker.feed(data)
    for m in self._unpacker:
        self._dispatch_message(m, disp_table)