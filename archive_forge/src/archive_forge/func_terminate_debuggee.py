import os
import subprocess
import sys
from debugpy import adapter, common
from debugpy.common import log, messaging, sockets
from debugpy.adapter import components, servers, sessions
def terminate_debuggee(self):
    with self.session:
        if self.exit_code is None:
            try:
                self.channel.request('terminate')
            except Exception:
                pass