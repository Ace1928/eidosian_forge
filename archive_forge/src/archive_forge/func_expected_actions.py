import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util
@classmethod
def expected_actions(cls):
    return [('action-1', 'description'), ('action-2', ''), flight.ActionType('action-3', 'more detail')]