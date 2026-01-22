import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
def proc_time(s):
    return time.time() - s['Start Time']