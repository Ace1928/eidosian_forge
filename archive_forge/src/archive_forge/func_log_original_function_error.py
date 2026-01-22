import warnings
from mlflow.utils.autologging_utils import _logger
def log_original_function_error(self, session, patch_obj, function_name, call_args, call_kwargs, exception):
    """Called during the execution of a patched API associated with an autologging integration
        when the original / underlying API invocation terminates with an error. For example,
        when a patched implementation of `sklearn.linear_model.LogisticRegression.fit()` invokes the
        original / underlying implementation of `LogisticRegression.fit()`, then this function is
        called if the original / underlying implementation terminates with an exception.

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the original API was called.
            function_name: The name of the original API that was called.
            call_args: The positional arguments passed to the original API call.
            call_kwargs: The keyword arguments passed to the original API call.
            exception: The exception that caused the original API call to terminate.
        """
    _logger.debug("Original function invocation threw exception during execution of patched API call '%s.%s' for %s autologging. Original function was invoked with args '%s' and kwargs '%s'. Exception: %s", patch_obj, function_name, session.integration, call_args, call_kwargs, exception)