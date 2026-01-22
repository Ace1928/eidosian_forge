import contextlib
import logging
import os
import sys
from oslo_utils import uuidutils
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils as eu  # noqa
def print_task_states(flowdetail, msg):
    eu.print_wrapped(msg)
    print("Flow '%s' state: %s" % (flowdetail.name, flowdetail.state))
    items = sorted(((td.name, td.version, td.state, td.results) for td in flowdetail))
    for item in items:
        print(' %s==%s: %s, result=%s' % item)