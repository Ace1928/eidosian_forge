import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
def kill_process_tree(process, use_psutil=True):
    """Terminate process and its descendants with SIGKILL"""
    if use_psutil and psutil is not None:
        _kill_process_tree_with_psutil(process)
    else:
        _kill_process_tree_without_psutil(process)