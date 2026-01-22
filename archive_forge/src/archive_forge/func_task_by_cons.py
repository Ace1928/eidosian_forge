from celery import _state
from celery._state import app_or_default, disable_trace, enable_trace, pop_current_task, push_current_task
from celery.local import Proxy
from .base import Celery
from .utils import AppPickler
def task_by_cons():
    app = _state.get_current_app()
    return app.tasks[name or app.gen_task_name(fun.__name__, fun.__module__)]