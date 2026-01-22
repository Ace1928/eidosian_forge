import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def cache_wrap_me(self, *args, **kwargs):
    cache_key = str((cb.__name__, *args, *kwargs.keys(), *kwargs.values()))
    if cache_key in self.cache_me:
        return self.cache_me[cache_key]
    ccb = cb(self, *args, **kwargs)
    self.cache_me[cache_key] = ccb
    return ccb