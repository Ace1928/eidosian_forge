import copy
import logging
import operator
import threading
import time
import traceback
from typing import Any, Dict, Optional
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.util import hash_launch_conf
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.tags import (
Collects launch data from queue populated by StandardAutoscaler.
        Launches nodes in a background thread.

        Overrides threading.Thread.run().
        NodeLauncher.start() executes this loop in a background thread.
        