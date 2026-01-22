import contextlib
import functools
import logging
import os
import sys
import time
import traceback
from kazoo import client
from taskflow.conductors import backends as conductor_backends
from taskflow import engines
from taskflow.jobs import backends as job_backends
from taskflow import logging as taskflow_logging
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends as persistence_backends
from taskflow.persistence import models
from taskflow import task
from oslo_utils import timeutils
from oslo_utils import uuidutils
def make_bottles(count):
    s = lf.Flow('bottle-song')
    take_bottle = TakeABottleDown('take-bottle-%s' % count, inject={'bottles_left': count}, provides='bottles_left')
    pass_it = PassItAround('pass-%s-around' % count)
    next_bottles = Conclusion('next-bottles-%s' % (count - 1))
    s.add(take_bottle, pass_it, next_bottles)
    for bottle in reversed(list(range(1, count))):
        take_bottle = TakeABottleDown('take-bottle-%s' % bottle, provides='bottles_left')
        pass_it = PassItAround('pass-%s-around' % bottle)
        next_bottles = Conclusion('next-bottles-%s' % (bottle - 1))
        s.add(take_bottle, pass_it, next_bottles)
    return s