from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
def get_rel_for_model(self, model):
    if isinstance(model, ModelAlias):
        model = model.model
    forwardrefs = self.model_refs.get(model, [])
    backrefs = self.model_backrefs.get(model, [])
    return (forwardrefs, backrefs)