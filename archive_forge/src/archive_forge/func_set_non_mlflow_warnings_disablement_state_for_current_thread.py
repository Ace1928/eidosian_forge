import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils
def set_non_mlflow_warnings_disablement_state_for_current_thread(self, disabled=True):
    """Disables (or re-enables) non-MLflow warnings for the current thread.

        Args:
            disabled: If `True`, disables non-MLflow warnings for the current thread. If `False`,
                enables non-MLflow warnings for the current thread. non-MLflow warning
                behavior in other threads is unaffected.

        """
    with self._state_lock:
        if disabled:
            self._disabled_threads.add(get_current_thread_id())
        else:
            self._disabled_threads.discard(get_current_thread_id())
        self._modify_patch_state_if_necessary()