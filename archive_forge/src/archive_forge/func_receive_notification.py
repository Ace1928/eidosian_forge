from collections import deque
import select
import msgpack
def receive_notification(self):
    """wait for the next incoming message.
        intended to be used when we have nothing to send but want to receive
        notifications.
        """
    if not self._endpoint.receive_messages():
        raise EOFError('EOF')
    self._process_input_notification()
    self._process_input_request()