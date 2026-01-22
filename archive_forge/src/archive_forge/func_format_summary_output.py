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
def format_summary_output(state_data: Dict, *, resource: StateResource) -> str:
    if len(state_data) == 0:
        return 'No resource in the cluster'
    cluster_data = state_data['cluster']
    summaries = cluster_data['summary']
    summary_by = cluster_data['summary_by']
    del cluster_data['summary_by']
    del cluster_data['summary']
    cluster_info_table = yaml.dump(cluster_data, indent=2)
    table = []
    headers = []
    for summary in summaries.values():
        for key, val in summary.items():
            if isinstance(val, dict):
                summary[key] = yaml.dump(val, indent=2)
        headers = sorted([key.upper() for key in summary.keys()])
        table.append([summary[header.lower()] for header in headers])
    summary_table = tabulate(table, headers=headers, showindex=True, tablefmt='plain', numalign='left')
    time = datetime.now()
    header = '=' * 8 + f' {resource.value.capitalize()} Summary: {time} ' + '=' * 8
    return f'\n{header}\nStats:\n------------------------------------\n{cluster_info_table}\n\nTable (group by {summary_by}):\n------------------------------------\n{summary_table}\n'