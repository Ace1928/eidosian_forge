def start_child_workflow_execution(self, workflow_type_name, workflow_type_version, workflow_id, child_policy=None, control=None, execution_start_to_close_timeout=None, input=None, tag_list=None, task_list=None, task_start_to_close_timeout=None):
    """
        Requests that a child workflow execution be started and
        records a StartChildWorkflowExecutionInitiated event in the
        history.  The child workflow execution is a separate workflow
        execution with its own history.
        """
    o = {}
    o['decisionType'] = 'StartChildWorkflowExecution'
    attrs = o['startChildWorkflowExecutionDecisionAttributes'] = {}
    attrs['workflowType'] = {'name': workflow_type_name, 'version': workflow_type_version}
    attrs['workflowId'] = workflow_id
    if child_policy is not None:
        attrs['childPolicy'] = child_policy
    if control is not None:
        attrs['control'] = control
    if execution_start_to_close_timeout is not None:
        attrs['executionStartToCloseTimeout'] = execution_start_to_close_timeout
    if input is not None:
        attrs['input'] = input
    if tag_list is not None:
        attrs['tagList'] = tag_list
    if task_list is not None:
        attrs['taskList'] = {'name': task_list}
    if task_start_to_close_timeout is not None:
        attrs['taskStartToCloseTimeout'] = task_start_to_close_timeout
    self._data.append(o)