from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def pytest_runtest_logreport(report):
    global _current_report
    if report.when == 'call':
        _current_report = report