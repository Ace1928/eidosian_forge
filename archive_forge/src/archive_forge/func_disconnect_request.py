from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
@message_handler
def disconnect_request(self, request):
    self.restart_requested = False
    terminate_debuggee = request('terminateDebuggee', bool, optional=True)
    if terminate_debuggee == ():
        terminate_debuggee = None
    self.session.finalize('client requested "disconnect"', terminate_debuggee)
    request.respond({})
    if self.using_stdio:
        servers.stop_serving()
        log.info('{0} disconnected from stdio; closing remaining server connections.', self)
        for conn in servers.connections():
            try:
                conn.channel.close()
            except Exception:
                log.swallow_exception()