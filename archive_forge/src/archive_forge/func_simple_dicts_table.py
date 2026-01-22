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
def simple_dicts_table():
    dict_values = pa.array(['foo', 'baz', 'quux'], type=pa.utf8())
    data = [pa.chunked_array([pa.DictionaryArray.from_arrays([1, 0, None], dict_values), pa.DictionaryArray.from_arrays([2, 1], dict_values)])]
    return pa.Table.from_arrays(data, names=['some_dicts'])