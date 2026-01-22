import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.types import notifier
import example_utils as eu  # noqa
def install_windows(frame, doors):
    return True