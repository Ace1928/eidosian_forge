import itertools
import os
import signal
import threading
import time
from debugpy import common
from debugpy.common import log, util
from debugpy.adapter import components, launchers, servers
def report_sockets():
    if not _sessions:
        return
    session = sorted(_sessions, key=lambda session: session.id)[0]
    client = session.client
    if client is not None:
        client.report_sockets()