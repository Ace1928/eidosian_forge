import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def get_demand_report(lm_summary: LoadMetricsSummary):
    demand_lines = []
    if lm_summary.resource_demand:
        demand_lines.extend(format_resource_demand_summary(lm_summary.resource_demand))
    for entry in lm_summary.pg_demand:
        pg, count = entry
        pg_str = format_pg(pg)
        line = f' {pg_str}: {count}+ pending placement groups'
        demand_lines.append(line)
    for bundle, count in lm_summary.request_demand:
        line = f' {bundle}: {count}+ from request_resources()'
        demand_lines.append(line)
    if len(demand_lines) > 0:
        demand_report = '\n'.join(demand_lines)
    else:
        demand_report = ' (no resource demands)'
    return demand_report