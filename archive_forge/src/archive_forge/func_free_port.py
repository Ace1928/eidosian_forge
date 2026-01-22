import socket
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
from selenium.types import AnyKey
from selenium.webdriver.common.keys import Keys
def free_port() -> int:
    """Determines a free port using sockets."""
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(('127.0.0.1', 0))
    free_socket.listen(5)
    port: int = free_socket.getsockname()[1]
    free_socket.close()
    return port