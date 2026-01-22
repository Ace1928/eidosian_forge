import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
from triad.utils.convert import (
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
def to_taskspec(obj: Any, parent_workflow_spec: Optional[WorkflowSpec]=None) -> TaskSpec:
    assert_arg_not_none(obj, 'obj')
    if isinstance(obj, str):
        return to_taskspec(json.loads(obj))
    if isinstance(obj, TaskSpec):
        return obj
    if isinstance(obj, Dict):
        d: Dict[str, Any] = dict(obj)
        node_spec: Optional[_NodeSpec] = None
        if 'node_spec' in d:
            aot(parent_workflow_spec is not None, lambda: InvalidOperationError('parent workflow must be set'))
            node_spec = _NodeSpec(workflow=parent_workflow_spec, **d['node_spec'])
            del d['node_spec']
        if 'tasks' in d:
            ts: TaskSpec = WorkflowSpec(**d)
        else:
            ts = TaskSpec(**d)
        if node_spec is not None:
            ts._node_spec = node_spec
        return ts
    raise TypeError(f"can't convert {obj} to TaskSpec")