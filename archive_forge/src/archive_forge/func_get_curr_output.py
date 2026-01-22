from _pydev_runfiles import pydev_runfiles_xml_rpc
import pickle
import zlib
import base64
import os
from pydevd_file_utils import canonical_normalized_path
import pytest
import sys
import time
from pathlib import Path
def get_curr_output():
    buf_out = State.buf_out
    buf_err = State.buf_err
    return (buf_out.getvalue() if buf_out is not None else '', buf_err.getvalue() if buf_err is not None else '')