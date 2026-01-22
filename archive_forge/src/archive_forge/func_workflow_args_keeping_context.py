import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef
@contextlib.contextmanager
def workflow_args_keeping_context() -> None:
    """
    This context only read workflow arguments. Workflows inside
    are untouched and can be serialized again properly.
    """
    global _resolve_workflow_refs
    _resolve_workflow_refs_bak = _resolve_workflow_refs

    def _keep_workflow_refs(index: int):
        return _KeepWorkflowRefs(index)
    _resolve_workflow_refs = _keep_workflow_refs
    try:
        yield
    finally:
        _resolve_workflow_refs = _resolve_workflow_refs_bak