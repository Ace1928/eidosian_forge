from collections import deque
import select
import msgpack
def process_incoming(self):
    self.receive_messages(all=True)