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
def create_sequences(self):
    if self.database.sequences:
        for field in self.model._meta.sorted_fields:
            if field.sequence:
                self.create_sequence(field)