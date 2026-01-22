from __future__ import annotations
import json
from typing import (
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def flatten_run(self, run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Utility to flatten a nest run object into a list of runs.
        :param run: The base run to flatten.
        :return: The flattened list of runs.
        """

    def flatten(child_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Utility to recursively flatten a list of child runs in a run.
            :param child_runs: The list of child runs to flatten.
            :return: The flattened list of runs.
            """
        if child_runs is None:
            return []
        result = []
        for item in child_runs:
            child_runs = item.pop('child_runs', [])
            result.append(item)
            result.extend(flatten(child_runs))
        return result
    return flatten([run])