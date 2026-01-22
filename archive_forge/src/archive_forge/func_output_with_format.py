import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
def output_with_format(state_data: List[Dict], *, schema: Optional[StateSchema], format: AvailableFormat=AvailableFormat.DEFAULT, detail: bool=False) -> str:
    if schema:
        state_data = [schema.humanify(state) for state in state_data]
    if format == AvailableFormat.DEFAULT:
        return get_table_output(state_data, schema, detail)
    if format == AvailableFormat.YAML:
        return yaml.dump(state_data, indent=4, explicit_start=True, sort_keys=False, explicit_end=True)
    elif format == AvailableFormat.JSON:
        return json.dumps(state_data)
    elif format == AvailableFormat.TABLE:
        return get_table_output(state_data, schema, detail)
    else:
        raise ValueError(f'Unexpected format: {format}. Supported formatting: {_get_available_formats()}')