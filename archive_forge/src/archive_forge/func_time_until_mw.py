import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
@property
def time_until_mw(self):
    """
        Returns the time until the next Maintenance Window as a
        datetime.timedelta object.
        """
    return self._get_time_until_mw()