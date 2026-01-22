import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread
from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, \
from jedi.api.exceptions import InternalError
def get_or_create_access_handle(self, obj):
    id_ = id(obj)
    try:
        return self.get_access_handle(id_)
    except KeyError:
        access = DirectObjectAccess(self._inference_state_weakref(), obj)
        handle = AccessHandle(self, access, id_)
        self.set_access_handle(handle)
        return handle