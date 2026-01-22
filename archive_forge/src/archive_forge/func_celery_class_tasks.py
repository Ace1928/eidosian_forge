import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union  # noqa
import pytest
@pytest.fixture(scope='session')
def celery_class_tasks():
    """Redefine this fixture to register tasks with the test Celery app."""
    return []