from collections import namedtuple
import logging
import json
from typing import Optional
from ray.tune.execution.placement_groups import (
from ray.tune.utils.resource_updater import _Resources
from ray.util.annotations import Deprecated, DeveloperAPI
from ray.tune import TuneError
@DeveloperAPI
def json_to_resources(data: Optional[str]) -> Optional[PlacementGroupFactory]:
    if data is None or data == 'null':
        return None
    if isinstance(data, str):
        data = json.loads(data)
    for k in data:
        if k in ['driver_cpu_limit', 'driver_gpu_limit']:
            raise TuneError('The field `{}` is no longer supported. Use `extra_cpu` or `extra_gpu` instead.'.format(k))
        if k not in _Resources._fields:
            raise ValueError('Unknown resource field {}, must be one of {}'.format(k, Resources._fields))
    resource_dict_to_pg_factory(dict(cpu=data.get('cpu', 1), gpu=data.get('gpu', 0), memory=data.get('memory', 0), custom_resources=data.get('custom_resources')))