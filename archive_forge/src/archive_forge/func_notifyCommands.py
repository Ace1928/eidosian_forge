import unittest
from _pydev_bundle._pydev_saved_modules import thread
import queue as Queue
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
import os
import threading
import sys
def notifyCommands(self, job_id, commands):
    for command in commands:
        getattr(self, command[0])(job_id, *command[1], **command[2])
    return True