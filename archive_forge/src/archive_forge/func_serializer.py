import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef
def serializer(w):
    if w in deduplicator:
        return deduplicator[w]
    if isinstance(w, WorkflowRef):
        w.ref = None
    i = len(workflow_refs)
    workflow_refs.append(w)
    deduplicator[w] = i
    return i