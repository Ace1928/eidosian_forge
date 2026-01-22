import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def get_parse_out_err(p):
    return [b.splitlines() for b in p.communicate()]