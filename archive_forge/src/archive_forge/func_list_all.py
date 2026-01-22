import functools
import logging
from typing import Dict, Set, List, Tuple, Union, Optional, Any
import time
import uuid
import ray
from ray.dag import DAGNode
from ray.dag.input_node import DAGInputData
from ray.remote_function import RemoteFunction
from ray.workflow.common import (
from ray.workflow import serialization, workflow_access, workflow_context
from ray.workflow.event_listener import EventListener, EventListenerType, TimerListener
from ray.workflow.workflow_storage import WorkflowStorage
from ray.workflow.workflow_state_from_dag import workflow_state_from_dag
from ray.util.annotations import PublicAPI
from ray._private.usage import usage_lib
@PublicAPI(stability='alpha')
@client_mode_wrap
def list_all(status_filter: Optional[Union[Union[WorkflowStatus, str], Set[Union[WorkflowStatus, str]]]]=None) -> List[Tuple[str, WorkflowStatus]]:
    """List all workflows matching a given status filter. When returning "RESUMEABLE"
    workflows, the workflows that was running ranks before the workflow that was pending
    in the result list.

    Args:
        status_filter: If given, only returns workflow with that status. This can
            be a single status or set of statuses. The string form of the
            status is also acceptable, i.e.,
            "RUNNING"/"FAILED"/"SUCCESSFUL"/"CANCELED"/"RESUMABLE"/"PENDING".

    Examples:
        .. testcode::

            from ray import workflow

            @ray.remote
            def long_running_job():
                import time
                time.sleep(2)

            workflow_task = long_running_job.bind()
            wf = workflow.run_async(workflow_task,
                workflow_id="long_running_job")
            jobs = workflow.list_all(workflow.RUNNING)
            assert jobs == [ ("long_running_job", workflow.RUNNING) ]
            ray.get(wf)
            jobs = workflow.list_all({workflow.RUNNING})
            assert jobs == []

    Returns:
        A list of tuple with workflow id and workflow status
    """
    _ensure_workflow_initialized()
    if isinstance(status_filter, str):
        status_filter = set({WorkflowStatus(status_filter)})
    elif isinstance(status_filter, WorkflowStatus):
        status_filter = set({status_filter})
    elif isinstance(status_filter, set):
        if all((isinstance(s, str) for s in status_filter)):
            status_filter = {WorkflowStatus(s) for s in status_filter}
        elif not all((isinstance(s, WorkflowStatus) for s in status_filter)):
            raise TypeError(f'status_filter contains element which is not a type of `WorkflowStatus or str`. {status_filter}')
    elif status_filter is None:
        status_filter = set(WorkflowStatus)
        status_filter.discard(WorkflowStatus.NONE)
    else:
        raise TypeError('status_filter must be WorkflowStatus or a set of WorkflowStatus.')
    try:
        workflow_manager = workflow_access.get_management_actor()
    except ValueError:
        workflow_manager = None
    if workflow_manager is None:
        non_terminating_workflows = {}
    else:
        non_terminating_workflows = ray.get(workflow_manager.list_non_terminating_workflows.remote())
    ret = []
    if set(non_terminating_workflows.keys()).issuperset(status_filter):
        for status, workflows in non_terminating_workflows.items():
            if status in status_filter:
                for w in workflows:
                    ret.append((w, status))
        return ret
    ret = []
    store = WorkflowStorage('')
    modified_status_filter = status_filter.copy()
    modified_status_filter.update(WorkflowStatus.non_terminating_status())
    status_from_storage = store.list_workflow(modified_status_filter)
    non_terminating_workflows = {k: set(v) for k, v in non_terminating_workflows.items()}
    resume_running = []
    resume_pending = []
    for k, s in status_from_storage:
        if s in non_terminating_workflows and k not in non_terminating_workflows[s]:
            if s == WorkflowStatus.RUNNING:
                resume_running.append(k)
            elif s == WorkflowStatus.PENDING:
                resume_pending.append(k)
            else:
                assert False, 'This line of code should not be reachable.'
            continue
        if s in status_filter:
            ret.append((k, s))
    if WorkflowStatus.RESUMABLE in status_filter:
        for w in resume_running:
            ret.append((w, WorkflowStatus.RESUMABLE))
        for w in resume_pending:
            ret.append((w, WorkflowStatus.RESUMABLE))
    return ret