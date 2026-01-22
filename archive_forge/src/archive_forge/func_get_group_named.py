import errno
import glob
import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import logging
import os
import os.path
import struct
import sys
def get_group_named(self, group, path=None):
    result = {}
    for ep in self.get_group_all(group, path=path):
        if ep.name not in result:
            result[ep.name] = ep
    return result