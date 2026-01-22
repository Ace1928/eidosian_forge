from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from six.moves import queue
Adds an item to the buffer.

    Args:
      task (Union[task.Task, str]): A buffered item. Expected to be a task or a
        string (to handle shutdowns) when used by task_graph_executor.
      prioritize (bool): Tasks added with prioritize=True will be returned by
        `get` before tasks added with prioritize=False.
    