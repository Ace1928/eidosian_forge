import logging
import os
import pathlib
import sys
import time
import pytest
def is_stderr_output_handled():
    from kivy.logger import LoggerHistory
    LoggerHistory.clear_history()
    sys.stderr.write('Test output to stderr\n')
    sys.stderr.flush()
    return bool(LoggerHistory.history)