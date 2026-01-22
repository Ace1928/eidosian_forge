from __future__ import print_function
import logging
import os
import sys
import threading
import time
import subprocess
from wsgiref.simple_server import WSGIRequestHandler
from pecan.commands import BaseCommand
from pecan import util
def should_reload(self, event):
    for t in (FileSystemMovedEvent, FileModifiedEvent, DirModifiedEvent):
        if isinstance(event, t):
            return True
    return False