import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
Waits for the :py:meth:`.start` method to gracefully exit.

        An optional timeout can be provided, which will cause the method to
        return within the specified timeout. If the timeout is reached, the
        returned value will be False.

        :param timeout: Maximum number of seconds that the :meth:`.wait`
                        method should block for
        :type timeout: float/int
        