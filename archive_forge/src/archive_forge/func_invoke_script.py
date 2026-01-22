import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def invoke_script(script_name, *args):
    subprocess_env = get_modified_env_with_pythonpath()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    python_file = os.path.join(dir_path, script_name)
    cmd = [sys.executable, python_file]
    cmd.extend(args)
    subprocess.check_call(cmd, env=subprocess_env)