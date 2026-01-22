import copy
from oslo_serialization import jsonutils
import yaml
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import signal_responder
from heat.engine import support
from heat.engine import translation
def build_tasks(self, props):
    for task in props[self.TASKS]:
        current_task = {}
        wf_value = task.get(self.WORKFLOW)
        if wf_value is not None:
            current_task.update({self.WORKFLOW: wf_value})
        if task.get(self.POLICIES) is not None:
            task.update(task.get(self.POLICIES))
        task_keys = [key for key in self._TASKS_KEYS if key not in [self.WORKFLOW, self.TASK_NAME, self.POLICIES]]
        for task_prop in task_keys:
            if task.get(task_prop) is not None:
                current_task.update({task_prop.replace('_', '-'): task[task_prop]})
        yield {task[self.TASK_NAME]: current_task}