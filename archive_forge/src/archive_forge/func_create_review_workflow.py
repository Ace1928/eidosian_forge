import contextlib
import itertools
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from zake import fake_client
from taskflow.conductors import backends as conductors
from taskflow import engines
from taskflow.jobs import backends as boards
from taskflow.patterns import linear_flow
from taskflow.persistence import backends as persistence
from taskflow.persistence import models
from taskflow import task
from taskflow.utils import threading_utils
def create_review_workflow():
    """Factory method used to create a review workflow to run."""
    f = linear_flow.Flow('tester')
    f.add(MakeTempDir(name='maker'), RunReview(name='runner'), CleanResources(name='cleaner'))
    return f