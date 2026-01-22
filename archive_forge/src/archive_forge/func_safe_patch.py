import abc
import functools
import inspect
import itertools
import typing
import uuid
from abc import abstractmethod
from contextlib import contextmanager
import mlflow
import mlflow.utils.autologging_utils
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.tracking.client import MlflowClient
from mlflow.utils import gorilla, is_iterator
from mlflow.utils.autologging_utils import _logger
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
def safe_patch(autologging_integration, destination, function_name, patch_function, manage_run=False, extra_tags=None):
    """Patches the specified `function_name` on the specified `destination` class for autologging
    purposes, preceding its implementation with an error-safe copy of the specified patch
    `patch_function` with the following error handling behavior:
        - Exceptions thrown from the underlying / original function
          (`<destination>.<function_name>`) are propagated to the caller.
        - Exceptions thrown from other parts of the patched implementation (`patch_function`)
          are caught and logged as warnings.

    Args:
        autologging_integration: The name of the autologging integration associated with the
            patch.
        destination: The Python class on which the patch is being defined.
        function_name: The name of the function to patch on the specified `destination` class.
        patch_function: The patched function code to apply. This is either a `PatchFunction`
            class definition or a function object. If it is a function object, the
            first argument should be reserved for an `original` method argument
            representing the underlying / original function. Subsequent arguments
            should be identical to those of the original function being patched.
        manage_run: If `True`, applies the `with_managed_run` wrapper to the specified
            `patch_function`, which automatically creates & terminates an MLflow
            active run during patch code execution if necessary. If `False`,
            does not apply the `with_managed_run` wrapper to the specified
            `patch_function`.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """
    from mlflow.utils.autologging_utils import autologging_is_disabled, get_autologging_config
    if manage_run:
        tags = _resolve_extra_tags(autologging_integration, extra_tags)
        patch_function = with_managed_run(autologging_integration, patch_function, tags=tags)
    patch_is_class = inspect.isclass(patch_function)
    if patch_is_class:
        assert issubclass(patch_function, PatchFunction)
    else:
        assert callable(patch_function)
    original_fn = gorilla.get_original_attribute(destination, function_name, bypass_descriptor_protocol=False)
    raw_original_obj = gorilla.get_original_attribute(destination, function_name, bypass_descriptor_protocol=True)
    if original_fn != raw_original_obj:
        raise RuntimeError(f'Unsupport patch on {destination}.{function_name}')
    elif isinstance(original_fn, property):
        is_property_method = True

        def original(self, *args, **kwargs):
            bound_delegate_method = original_fn.fget(self)
            return bound_delegate_method(*args, **kwargs)
    else:
        original = original_fn
        is_property_method = False

    def safe_patch_function(*args, **kwargs):
        """
        A safe wrapper around the specified `patch_function` implementation designed to
        handle exceptions thrown during the execution of `patch_function`. This wrapper
        distinguishes exceptions thrown from the underlying / original function
        (`<destination>.<function_name>`) from exceptions thrown from other parts of
        `patch_function`. This distinction is made by passing an augmented version of the
        underlying / original function to `patch_function` that uses nonlocal state to track
        whether or not it has been executed and whether or not it threw an exception.
        Exceptions thrown from the underlying / original function are propagated to the caller,
        while exceptions thrown from other parts of `patch_function` are caught and logged as
        warnings.
        """
        is_silent_mode = get_autologging_config(autologging_integration, 'silent', False)
        with set_mlflow_events_and_warnings_behavior_globally(reroute_warnings=True, disable_event_logs=is_silent_mode, disable_warnings=is_silent_mode), set_non_mlflow_warnings_behavior_for_current_thread(reroute_warnings=True, disable_warnings=is_silent_mode):
            if is_testing():
                preexisting_run_for_testing = mlflow.active_run()
            exclusive = get_autologging_config(autologging_integration, 'exclusive', False)
            user_created_fluent_run_is_active = mlflow.active_run() and (not _AutologgingSessionManager.active_session())
            active_session_failed = _AutologgingSessionManager.active_session() is not None and _AutologgingSessionManager.active_session().state == 'failed'
            if active_session_failed or autologging_is_disabled(autologging_integration) or (user_created_fluent_run_is_active and exclusive) or mlflow.utils.autologging_utils._AUTOLOGGING_GLOBALLY_DISABLED:
                with set_non_mlflow_warnings_behavior_for_current_thread(disable_warnings=False, reroute_warnings=False):
                    return original(*args, **kwargs)
            original_has_been_called = False
            original_result = None
            failed_during_original = False
            patch_function_run_for_testing = None
            patch_function_exception = None

            def try_log_autologging_event(log_fn, *args):
                try:
                    log_fn(*args)
                except Exception as e:
                    _logger.debug("Failed to log autologging event via '%s'. Exception: %s", log_fn, e)

            def call_original_fn_with_event_logging(original_fn, og_args, og_kwargs):
                try:
                    try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_start, session, destination, function_name, og_args, og_kwargs)
                    original_fn_result = original_fn(*og_args, **og_kwargs)
                    try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_success, session, destination, function_name, og_args, og_kwargs)
                    return original_fn_result
                except Exception as original_fn_e:
                    try_log_autologging_event(AutologgingEventLogger.get_logger().log_original_function_error, session, destination, function_name, og_args, og_kwargs, original_fn_e)
                    nonlocal failed_during_original
                    failed_during_original = True
                    raise
            with _AutologgingSessionManager.start_session(autologging_integration) as session:
                try:

                    def call_original(*og_args, **og_kwargs):

                        def _original_fn(*_og_args, **_og_kwargs):
                            if is_testing():
                                _validate_args(autologging_integration, function_name, args, kwargs, og_args, og_kwargs)
                                nonlocal patch_function_run_for_testing
                                patch_function_run_for_testing = mlflow.active_run()
                            nonlocal original_has_been_called
                            original_has_been_called = True
                            nonlocal original_result
                            with set_non_mlflow_warnings_behavior_for_current_thread(disable_warnings=False, reroute_warnings=False):
                                original_result = original(*_og_args, **_og_kwargs)
                                return original_result
                        return call_original_fn_with_event_logging(_original_fn, og_args, og_kwargs)
                    call_original = update_wrapper_extended(call_original, original)
                    try_log_autologging_event(AutologgingEventLogger.get_logger().log_patch_function_start, session, destination, function_name, args, kwargs)
                    if patch_is_class:
                        patch_function.call(call_original, *args, **kwargs)
                    else:
                        patch_function(call_original, *args, **kwargs)
                    session.state = 'succeeded'
                    try_log_autologging_event(AutologgingEventLogger.get_logger().log_patch_function_success, session, destination, function_name, args, kwargs)
                except Exception as e:
                    session.state = 'failed'
                    patch_function_exception = e
                    if failed_during_original or is_testing():
                        raise
                if is_testing() and (not preexisting_run_for_testing):
                    assert not mlflow.active_run(), f'Autologging integration {autologging_integration} leaked an active run'
                    if patch_function_run_for_testing:
                        _validate_autologging_run(autologging_integration, patch_function_run_for_testing.info.run_id)
                try:
                    if original_has_been_called:
                        return original_result
                    else:
                        return call_original_fn_with_event_logging(original, args, kwargs)
                finally:
                    if patch_function_exception is not None and (not failed_during_original):
                        try_log_autologging_event(AutologgingEventLogger.get_logger().log_patch_function_error, session, destination, function_name, args, kwargs, patch_function_exception)
                        _logger.warning('Encountered unexpected error during %s autologging: %s', autologging_integration, patch_function_exception)
    if is_property_method:

        def get_bound_safe_patch_fn(self):
            original_fn.fget(self)

            def bound_safe_patch_fn(*args, **kwargs):
                return safe_patch_function(self, *args, **kwargs)
            return update_wrapper_extended(bound_safe_patch_fn, original_fn.fget)
        get_bound_safe_patch_fn = update_wrapper_extended(get_bound_safe_patch_fn, original_fn.fget)
        safe_patch_obj = property(get_bound_safe_patch_fn)
    else:
        safe_patch_obj = update_wrapper_extended(safe_patch_function, original)
    new_patch = _wrap_patch(destination, function_name, safe_patch_obj)
    _store_patch(autologging_integration, new_patch)