import functools
from taskflow.engines.action_engine.actions import base
from taskflow import logging
from taskflow import states
from taskflow import task as task_atom
from taskflow.types import failure
def schedule_execution(self, task):
    self.change_state(task, states.RUNNING, progress=0.0)
    arguments = self._storage.fetch_mapped_args(task.rebind, atom_name=task.name, optional_args=task.optional)
    if task.notifier.can_be_registered(task_atom.EVENT_UPDATE_PROGRESS):
        progress_callback = functools.partial(self._on_update_progress, task)
    else:
        progress_callback = None
    task_uuid = self._storage.get_atom_uuid(task.name)
    return self._task_executor.execute_task(task, task_uuid, arguments, progress_callback=progress_callback)