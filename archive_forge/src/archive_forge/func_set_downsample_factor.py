import os
import time
from threading import Thread, Lock
import sentry_sdk
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
def set_downsample_factor(self):
    if self._healthy:
        if self._downsample_factor > 0:
            logger.debug('[Monitor] health check positive, reverting to normal sampling')
        self._downsample_factor = 0
    else:
        if self.downsample_factor < MAX_DOWNSAMPLE_FACTOR:
            self._downsample_factor += 1
        logger.debug('[Monitor] health check negative, downsampling with a factor of %d', self._downsample_factor)