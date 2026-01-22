import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef
@contextlib.contextmanager
def workflow_args_serialization_context(workflow_refs: List[WorkflowRef]) -> None:
    """
    This serialization context reduces workflow input arguments to three
    parts:

    1. A workflow input placeholder. It is an object without 'Workflow' and
       'ObjectRef' object. They are replaced with integer indices. During
       deserialization, we can refill the placeholder with a list of
       'Workflow' and a list of 'ObjectRef'. This provides us great
       flexibility, for example, during recovery we can plug an alternative
       list of 'Workflow' and 'ObjectRef', since we lose the original ones.
    2. A list of 'Workflow'. There is no duplication in it.
    3. A list of 'ObjectRef'. There is no duplication in it.

    We do not allow duplication because in the arguments duplicated workflows
    and object refs are shared by reference. So when deserialized, we also
    want them to be shared by reference. See
    "tests/test_object_deref.py:deref_shared" as an example.

    The deduplication works like this:
        Inputs: [A B A B C C A]
        Output List: [A B C]
        Index in placeholder: [0 1 0 1 2 2 0]

    Args:
        workflow_refs: Output list of workflows or references to workflows.
    """
    deduplicator: Dict[WorkflowRef, int] = {}

    def serializer(w):
        if w in deduplicator:
            return deduplicator[w]
        if isinstance(w, WorkflowRef):
            w.ref = None
        i = len(workflow_refs)
        workflow_refs.append(w)
        deduplicator[w] = i
        return i
    register_serializer(WorkflowRef, serializer=serializer, deserializer=_resolve_workflow_refs)
    try:
        yield
    finally:
        deregister_serializer(WorkflowRef)