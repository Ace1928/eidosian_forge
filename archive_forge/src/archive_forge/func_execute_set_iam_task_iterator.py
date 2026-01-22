from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks import task_util
def execute_set_iam_task_iterator(iterator, continue_on_error):
    """Executes single or multiple set-IAM tasks with different handling.

  Args:
    iterator (iter[set_iam_policy_task._SetIamPolicyTask]): Contains set IAM
      task(s) to execute.
    continue_on_error (bool): If multiple tasks in iterator, determines whether
      to continue executing after an error.

  Returns:
    int: Status code. For multiple tasks, the task executor will return if
      any of the tasks failed.
    object|None: If executing a single task, the newly set IAM policy. This
      is useful for outputting to the terminal.
  """
    plurality_checkable_task_iterator = plurality_checkable_iterator.PluralityCheckableIterator(iterator)
    if not plurality_checkable_task_iterator.is_plural():
        return (0, task_util.get_first_matching_message_payload(next(plurality_checkable_task_iterator).execute().messages, task.Topic.SET_IAM_POLICY))
    task_status_queue = task_graph_executor.multiprocessing_context.Queue()
    exit_code = task_executor.execute_tasks(plurality_checkable_task_iterator, parallelizable=True, task_status_queue=task_status_queue, progress_manager_args=task_status.ProgressManagerArgs(increment_type=task_status.IncrementType.INTEGER, manifest_path=None), continue_on_error=continue_on_error)
    return (exit_code, None)