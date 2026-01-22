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
def create_subprocess(self):
    self.server_process = subprocess.Popen([arg for arg in sys.argv if arg != '--reload'], stdout=sys.stdout, stderr=sys.stderr)