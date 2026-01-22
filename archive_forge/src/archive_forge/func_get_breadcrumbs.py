import json
from typing import Any, Callable, List
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.utils.input import get_bolded_text, get_colored_text
def get_breadcrumbs(self, run: Run) -> str:
    parents = self.get_parents(run)[::-1]
    string = ' > '.join((f'{parent.execution_order}:{parent.run_type}:{parent.name}' if i != len(parents) - 1 else f'{parent.execution_order}:{parent.run_type}:{parent.name}' for i, parent in enumerate(parents + [run])))
    return string