import os
import sys
import tempfile
from pathlib import Path
from shutil import rmtree
import pytest
def set_hsds_root():
    """Make required HSDS root directory."""
    hsds_root = Path(os.environ['ROOT_DIR']) / os.environ['BUCKET_NAME'] / 'home'
    if hsds_root.exists():
        rmtree(hsds_root)
    old_sysargv = sys.argv
    sys.argv = ['']
    sys.argv.extend(['-e', os.environ['HS_ENDPOINT']])
    sys.argv.extend(['-u', 'admin'])
    sys.argv.extend(['-p', 'admin'])
    sys.argv.extend(['--bucket', os.environ['BUCKET_NAME']])
    sys.argv.append('/home/')
    hstouch()
    sys.argv = ['']
    sys.argv.extend(['-e', os.environ['HS_ENDPOINT']])
    sys.argv.extend(['-u', 'admin'])
    sys.argv.extend(['-p', 'admin'])
    sys.argv.extend(['--bucket', os.environ['BUCKET_NAME']])
    sys.argv.extend(['-o', os.environ['HS_USERNAME']])
    sys.argv.append(f'/home/{os.environ['HS_USERNAME']}/')
    hstouch()
    sys.argv = old_sysargv