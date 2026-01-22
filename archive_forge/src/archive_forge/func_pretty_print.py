import abc
import json
import logging
import os
import pyarrow
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Type
import yaml
from ray.air._internal.json import SafeFallbackEncoder
from ray.tune.callback import Callback
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@DeveloperAPI
def pretty_print(result, exclude: Optional[Set[str]]=None):
    result = result.copy()
    result.update(config=None)
    result.update(hist_stats=None)
    out = {}
    for k, v in result.items():
        if v is not None and (exclude is None or k not in exclude):
            out[k] = v
    cleaned = json.dumps(out, cls=SafeFallbackEncoder)
    return yaml.dump(json.loads(cleaned), Dumper=_RayDumper, default_flow_style=False)