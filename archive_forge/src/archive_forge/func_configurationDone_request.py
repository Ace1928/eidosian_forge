from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
@message_handler
def configurationDone_request(self, request):
    if self.start_request is None or self.has_started:
        request.cant_handle('"configurationDone" is only allowed during handling of a "launch" or an "attach" request')
    try:
        self.has_started = True
        try:
            result = self.server.channel.delegate(request)
        except messaging.NoMoreMessages:
            request.respond({})
            self.start_request.respond({})
            self.session.finalize('{0} disconnected before responding to {1}'.format(self.server, json.repr(request.command)))
            return
        else:
            request.respond(result)
    except messaging.MessageHandlingError as exc:
        self.start_request.cant_handle(str(exc))
    finally:
        if self.start_request.response is None:
            self.start_request.respond({})
            self._propagate_deferred_events()
    for conn in servers.connections():
        if conn.server is None and conn.ppid == self.session.pid:
            self.notify_of_subprocess(conn)