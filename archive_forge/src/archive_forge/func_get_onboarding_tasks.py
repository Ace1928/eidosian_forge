from typing import Dict, List, Set, Any
import json
import os
import queue
import random
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.utils.misc import warn_once
def get_onboarding_tasks(self, worker_id: str) -> List[Dict[str, Any]]:
    """
        Get next onboarding task for given worker.

        :param worker_id:
            worker id

        :return:
            A list of onboarding tasks for the worker
        """
    if len(self.onboarding_tasks) == 0:
        return []
    worker_data = self._get_worker_data(worker_id)
    onboarding_todo = worker_data['onboarding_todo']
    if not onboarding_todo:
        return []
    num_tasks_to_return = min(len(onboarding_todo), self.opt['subtasks_per_hit'])
    onboarding_tasks_chosen = onboarding_todo[:num_tasks_to_return]
    worker_data['onboarding_todo'] = onboarding_todo[num_tasks_to_return:]
    return [self.onboarding_tasks[t_id] for t_id in onboarding_tasks_chosen]