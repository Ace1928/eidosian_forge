from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.react_task_demo.react_custom_no_extra_deps.worlds import (
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.tasks.react_task_demo.react_custom_no_extra_deps.task_config import (
import os
def check_workers_eligibility(workers):
    filled_roles = []
    use_workers = []
    for worker in workers:
        if worker.demo_role not in filled_roles:
            use_workers.append(worker)
            filled_roles.append(worker.demo_role)
    return use_workers