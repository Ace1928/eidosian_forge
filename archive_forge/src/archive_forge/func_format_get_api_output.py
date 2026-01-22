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
def format_get_api_output(state_data: Optional[StateSchema], id: str, *, schema: StateSchema, format: AvailableFormat=AvailableFormat.YAML) -> str:
    if not state_data or (isinstance(state_data, list) and len(state_data) == 0):
        return f'Resource with id={id} not found in the cluster.'
    if not isinstance(state_data, list):
        state_data = [state_data]
    state_data = [state.asdict() for state in state_data]
    return output_with_format(state_data, schema=schema, format=format, detail=True)