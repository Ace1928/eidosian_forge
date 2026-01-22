import copy
import enum
import functools
import logging
import multiprocessing
import shlex
import sys
import threading
from oslo_config import cfg
from oslo_config import types
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import daemon
This is intended to be used as a decorator with timeout.