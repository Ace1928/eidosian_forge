import os
import time
from threading import Thread, Lock
import sentry_sdk
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
@property
def downsample_factor(self):
    self._ensure_running()
    return self._downsample_factor