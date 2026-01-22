import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def fix_path(path, rel=None):
    path = os.path.join(outdir, path)
    dirname, basename = os.path.split(source)
    root, ext = os.path.splitext(basename)
    path = self.ExpandRuleVariables(path, root, dirname, source, ext, basename)
    if rel:
        path = os.path.relpath(path, rel)
    return path