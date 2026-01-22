import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
def bind_out(self, addr: str) -> None:
    """Enqueue ZMQ address for binding on out_socket.

        See zmq.Socket.bind for details.
        """
    self._out_binds.append(addr)