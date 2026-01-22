import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def test_release_unused():
    pool = pa.default_memory_pool()
    pool.release_unused()