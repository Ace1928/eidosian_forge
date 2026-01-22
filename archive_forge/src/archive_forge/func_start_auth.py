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
def start_auth(self, context: zmq.Context) -> bool:
    """
        Starts the ZMQ auth service thread, enabling authorization on all sockets within this context
        """
    if not self.auth_configured:
        return False
    self._socket.curve_secretkey = self._auth_config.server_secret_key
    self._socket.curve_publickey = self._auth_config.server_public_key
    self._socket.curve_server = True
    self._authenticator = AsyncioAuthenticator(context)
    if self._preloaded_keys:
        self.set_client_keys(self._preloaded_keys)
    else:
        self.load_client_keys_from_directory()
    self._authenticator.start()
    return True