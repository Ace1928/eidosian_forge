import copy
import os
import socket
import ssl
import sys
import threading
import weakref
from abc import abstractmethod
from itertools import chain
from queue import Empty, Full, LifoQueue
from time import time
from typing import Any, Callable, List, Optional, Type, Union
from urllib.parse import parse_qs, unquote, urlparse
from ._parsers import Encoder, _HiredisParser, _RESP2Parser, _RESP3Parser
from .backoff import NoBackoff
from .credentials import CredentialProvider, UsernamePasswordCredentialProvider
from .exceptions import (
from .retry import Retry
from .utils import (
def send_packed_command(self, command, check_health=True):
    """Send an already packed command to the Redis server"""
    if not self._sock:
        self.connect()
    if check_health:
        self.check_health()
    try:
        if isinstance(command, str):
            command = [command]
        for item in command:
            self._sock.sendall(item)
    except socket.timeout:
        self.disconnect()
        raise TimeoutError('Timeout writing to socket')
    except OSError as e:
        self.disconnect()
        if len(e.args) == 1:
            errno, errmsg = ('UNKNOWN', e.args[0])
        else:
            errno = e.args[0]
            errmsg = e.args[1]
        raise ConnectionError(f'Error {errno} while writing to socket. {errmsg}.')
    except BaseException:
        self.disconnect()
        raise