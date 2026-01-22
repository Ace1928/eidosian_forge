import warnings
from mlflow.utils.autologging_utils import _logger
def log_original_function_start(self, session, patch_obj, function_name, call_args, call_kwargs):
    """
        Called during the execution of a patched API associated with an autologging integration
        when the original / underlying API is invoked. For example, this is called when
        a patched implementation of `sklearn.linear_model.LogisticRegression.fit()` invokes
        the original implementation of `sklearn.linear_model.LogisticRegression.fit()`.

        Args:
            session: The `AutologgingSession` associated with the patched API call.
            patch_obj: The object (class, module, etc) on which the original API was called.
            function_name: The name of the original API that was called.
            call_args: The positional arguments passed to the original API call.
            call_kwargs: The keyword arguments passed to the original API call.
        """
    _logger.debug("Original function invoked during execution of patched API '%s.%s' for %s autologging. Original function was invoked with args '%s' and kwargs '%s'", patch_obj, function_name, session.integration, call_args, call_kwargs)