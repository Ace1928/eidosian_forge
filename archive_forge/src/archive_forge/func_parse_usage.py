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
def parse_usage(usage: Usage, verbose: bool) -> List[str]:
    placement_group_resource_usage = {}
    placement_group_resource_total = collections.defaultdict(float)
    for resource, (used, total) in usage.items():
        pg_resource_name, pg_name, is_countable = parse_placement_group_resource_str(resource)
        if pg_name:
            if pg_resource_name not in placement_group_resource_usage:
                placement_group_resource_usage[pg_resource_name] = 0
            if is_countable:
                placement_group_resource_usage[pg_resource_name] += used
                placement_group_resource_total[pg_resource_name] += total
            continue
    usage_lines = []
    for resource, (used, total) in sorted(usage.items()):
        if 'node:' in resource:
            continue
        _, pg_name, _ = parse_placement_group_resource_str(resource)
        if pg_name:
            continue
        pg_used = 0
        pg_total = 0
        used_in_pg = resource in placement_group_resource_usage
        if used_in_pg:
            pg_used = placement_group_resource_usage[resource]
            pg_total = placement_group_resource_total[resource]
            used = used - pg_total + pg_used
        if resource in ['memory', 'object_store_memory']:
            formatted_used = format_memory(used)
            formatted_total = format_memory(total)
            line = f'{formatted_used}/{formatted_total} {resource}'
            if used_in_pg:
                formatted_pg_used = format_memory(pg_used)
                formatted_pg_total = format_memory(pg_total)
                line = line + (f' ({formatted_pg_used} used of {formatted_pg_total} ' + 'reserved in placement groups)')
            usage_lines.append(line)
        elif resource.startswith('accelerator_type:') and (not verbose):
            pass
        else:
            line = f'{used}/{total} {resource}'
            if used_in_pg:
                line += f' ({pg_used} used of {pg_total} reserved in placement groups)'
            usage_lines.append(line)
    return usage_lines