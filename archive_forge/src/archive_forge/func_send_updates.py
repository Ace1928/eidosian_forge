from tornado.websocket import WebSocketHandler
import logging
from typing import Dict
@classmethod
def send_updates(cls: 'RequestInfoSocketHandler', msg: Dict) -> None:
    """Class method used to dispatch the request info to the waiting
        notebook. This method is called in `VoilaHandler` when the request
        info becomes available.
        If this method is called before the opening of websocket connection,
        `msg` is stored in `_cache0` and the message will be dispatched when
        a notebook with corresponding kernel id is connected.

        Args:
            - msg (Dict): this dictionary contains the `kernel_id` to identify
            the waiting notebook and `payload` is the request info.
        """
    kernel_id = msg['kernel_id']
    payload = msg['payload']
    waiter = cls._waiters.get(kernel_id, None)
    if waiter is not None:
        try:
            waiter.write_message(payload)
        except Exception:
            logging.error('Error sending message', exc_info=True)
    cls._cache[kernel_id] = payload