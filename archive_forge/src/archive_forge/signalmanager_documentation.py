from typing import Any, List, Tuple
from pydispatch import dispatcher
from twisted.internet.defer import Deferred
from scrapy.utils import signal as _signal

        Disconnect all receivers from the given signal.

        :param signal: the signal to disconnect from
        :type signal: object
        