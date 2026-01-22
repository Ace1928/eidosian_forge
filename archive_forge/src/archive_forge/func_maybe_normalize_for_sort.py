import copy
import logging
from functools import total_ordering
from tornado import web
from ..utils.tasks import as_dict, get_task_by_id, iter_tasks
from ..views import BaseHandler
@classmethod
def maybe_normalize_for_sort(cls, tasks, sort_by):
    sort_keys = {'name': str, 'state': str, 'received': float, 'started': float, 'runtime': float}
    if sort_by in sort_keys:
        for _, task in tasks:
            attr_value = getattr(task, sort_by, None)
            if attr_value:
                try:
                    setattr(task, sort_by, sort_keys[sort_by](attr_value))
                except TypeError:
                    pass