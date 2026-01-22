from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def environment_for_gdb():
    env = {}
    for var in ['PATH', 'LD_LIBRARY_PATH']:
        try:
            env[var] = os.environ[var]
        except KeyError:
            pass
    return env