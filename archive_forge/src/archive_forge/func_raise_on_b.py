import os
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
import pyarrow.tests.util as test_util
def raise_on_b(s):
    if s == 'b':
        raise ValueError('wtf')