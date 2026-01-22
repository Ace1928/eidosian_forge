import atexit
import errno
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from paste.deploy import loadapp, loadserver
from paste.script.command import Command, BadCommand
def loadserver(self, server_spec, name, relative_to, **kw):
    return loadserver(server_spec, name=name, relative_to=relative_to, **kw)