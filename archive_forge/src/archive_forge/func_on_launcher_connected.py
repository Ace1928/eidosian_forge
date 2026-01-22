import os
import subprocess
import sys
from debugpy import adapter, common
from debugpy.common import log, messaging, sockets
from debugpy.adapter import components, servers, sessions
def on_launcher_connected(sock):
    listener.close()
    stream = messaging.JsonIOStream.from_socket(sock)
    Launcher(session, stream)