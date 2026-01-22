import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.types import notifier
def task_watch(state, details):
    print('Task %s => %s' % (details.get('task_name'), state))