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
def random_ascii(length):
    return bytes(np.random.randint(65, 123, size=length, dtype='i1'))