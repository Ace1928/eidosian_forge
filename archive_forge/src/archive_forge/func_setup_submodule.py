from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import subprocess
import importlib
import pkg_resources
import threading
from subprocess import check_output
from dataclasses import dataclass
from typing import Optional
from fileio import File, PathIO, PathIOLike
from lazyops.envs import logger
from lazyops.envs import LazyEnv
def setup_submodule(self, name, *args, **kwargs):
    self.lock.acquire()
    with self.lock:
        if name not in LazyImporter.submodules:
            LazyImporter.submodules[name] = LazySubmodule(*args, name=name, **kwargs)
            LazyImporter.submodules[name].lazy_init()
        return LazyImporter.submodules[name]