import boto
from boto.services.message import ServiceMessage
from boto.services.servicedef import ServiceDef
from boto.pyami.scriptbase import ScriptBase
from boto.utils import get_ts
import time
import os
import mimetypes
def split_key(key):
    if key.find(';') < 0:
        t = (key, '')
    else:
        key, type = key.split(';')
        label, mtype = type.split('=')
        t = (key, mtype)
    return t