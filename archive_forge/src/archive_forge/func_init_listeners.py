import asyncio
import os
import sys
import traceback
from typing import Dict, Text, Tuple, cast
from jupyter_core.paths import jupyter_config_path
from jupyter_server.services.config import ConfigManager
from traitlets import Bool
from traitlets import Dict as Dict_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from .constants import (
from .schema import LANGUAGE_SERVER_SPEC_MAP
from .session import LanguageServerSession
from .trait_types import LoadableCallable, Schema
from .types import (
def init_listeners(self):
    """register traitlets-configured listeners"""
    scopes = {MessageScope.ALL: [self.all_listeners, EP_LISTENER_ALL_V1], MessageScope.CLIENT: [self.client_listeners, EP_LISTENER_CLIENT_V1], MessageScope.SERVER: [self.server_listeners, EP_LISTENER_SERVER_V1]}
    for scope, trt_ep in scopes.items():
        listeners, entry_point = trt_ep
        for ept in entry_points(group=entry_point):
            try:
                listeners.append(ept.load())
            except Exception as err:
                self.log.warning('Failed to load entry point %s: %s', ept.name, err)
        for listener in listeners:
            self.__class__.register_message_listener(scope=scope.value)(listener)