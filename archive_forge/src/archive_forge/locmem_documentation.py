import pickle
import time
from collections import OrderedDict
from threading import Lock
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
Thread-safe in-memory cache backend.