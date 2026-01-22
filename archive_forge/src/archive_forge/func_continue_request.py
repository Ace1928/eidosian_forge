from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
@message_handler
def continue_request(self, request):
    request.arguments['threadId'] = '*'
    try:
        return self.server.channel.delegate(request)
    except messaging.NoMoreMessages:
        return {'allThreadsContinued': True}