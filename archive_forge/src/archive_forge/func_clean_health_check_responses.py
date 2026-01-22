import copy
import re
import threading
import time
import warnings
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type, Union
from redis._parsers.encoders import Encoder
from redis._parsers.helpers import (
from redis.commands import (
from redis.connection import (
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def clean_health_check_responses(self) -> None:
    """
        If any health check responses are present, clean them
        """
    ttl = 10
    conn = self.connection
    while self.health_check_response_counter > 0 and ttl > 0:
        if self._execute(conn, conn.can_read, timeout=conn.socket_timeout):
            response = self._execute(conn, conn.read_response)
            if self.is_health_check_response(response):
                self.health_check_response_counter -= 1
            else:
                raise PubSubError('A non health check response was cleaned by execute_command: {0}'.format(response))
        ttl -= 1