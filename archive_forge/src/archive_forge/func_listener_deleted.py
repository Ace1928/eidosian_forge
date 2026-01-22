import contextlib
import logging
import threading
from threading import local as thread_local
from threading import Thread
import traceback
from types import MethodType
import weakref
import sys
from .constants import ComparisonMode, TraitKind
from .trait_base import Uninitialized
from .trait_errors import TraitNotificationError
def listener_deleted(self, ref):
    try:
        self.owner.remove(self)
    except ValueError:
        pass
    self.object = self.owner = None