import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def mark_active(self, ip):
    assert ip is not None, 'IP should be known at this time'
    logger.debug('Node {} is newly setup, treating as active'.format(ip))
    self.last_heartbeat_time_by_ip[ip] = time.time()