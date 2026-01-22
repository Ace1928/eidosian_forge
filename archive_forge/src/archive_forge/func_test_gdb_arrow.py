from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_gdb_arrow(gdb_arrow):
    s = gdb_arrow.print_value('42 + 1')
    assert s == '43'