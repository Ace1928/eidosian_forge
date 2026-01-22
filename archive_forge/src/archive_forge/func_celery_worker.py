import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture()
def celery_worker(request, celery_app, celery_includes, celery_worker_pool, celery_worker_parameters):
    """Fixture: Start worker in a thread, stop it when the test returns."""
    from .testing import worker
    if not NO_WORKER:
        for module in celery_includes:
            celery_app.loader.import_task_module(module)
        with worker.start_worker(celery_app, pool=celery_worker_pool, **celery_worker_parameters) as w:
            yield w