import logging
from typing import Any, Callable, Dict
def load_scheduler(scheduler_type: str) -> Any:
    scheduler_type = scheduler_type.lower()
    if scheduler_type not in _WANDB_SCHEDULERS:
        raise SchedulerError(f'The `scheduler_name` argument must be one of {list(_WANDB_SCHEDULERS.keys())}, got: {scheduler_type}')
    log.warn(f'Loading dependencies for Scheduler of type: {scheduler_type}')
    import_func = _WANDB_SCHEDULERS[scheduler_type]
    return import_func()