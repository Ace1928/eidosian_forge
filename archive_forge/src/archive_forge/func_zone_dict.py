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
def zone_dict(self):
    if self._zone_dict is None:
        zones = self.ex_list_zones()
        self._zone_dict = {zone.name: zone for zone in zones}
    return self._zone_dict