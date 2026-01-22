import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef
@contextlib.contextmanager
def workflow_args_resolving_context(workflow_ref_mapping: List[Any]) -> None:
    """
    This context resolves workflows and object refs inside workflow
    arguments into correct values.

    Args:
        workflow_ref_mapping: List of workflow refs.
    """
    global _resolve_workflow_refs
    _resolve_workflow_refs_bak = _resolve_workflow_refs
    _resolve_workflow_refs = workflow_ref_mapping.__getitem__
    try:
        yield
    finally:
        _resolve_workflow_refs = _resolve_workflow_refs_bak