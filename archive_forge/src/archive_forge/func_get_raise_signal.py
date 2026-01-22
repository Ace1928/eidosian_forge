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
def get_raise_signal():
    if sys.version_info >= (3, 8):
        return signal.raise_signal
    elif os.name == 'nt':
        pytest.skip('test requires Python 3.8+ on Windows')
    else:

        def raise_signal(signum):
            os.kill(os.getpid(), signum)
        return raise_signal