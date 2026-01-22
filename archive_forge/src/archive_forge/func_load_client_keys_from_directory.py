import asyncio
import logging
import sys
from asyncio import AbstractEventLoop
from typing import Callable, List, Optional, Tuple
from datetime import datetime
import zmq.asyncio
from zmq.auth.asyncio import AsyncioAuthenticator
from rpcq._base import to_msgpack, from_msgpack
from rpcq._spec import RPCSpec
from rpcq.messages import RPCRequest
def load_client_keys_from_directory(self, directory: Optional[str]=None) -> bool:
    """
        Reset authorized public key list to those present in the specified directory
        """
    if directory is None:
        if self._auth_config.client_keys_directory:
            directory = self._auth_config.client_keys_directory
    if not directory or not self.auth_configured:
        return False
    self._authenticator.configure_curve(domain='*', location=directory)
    return True