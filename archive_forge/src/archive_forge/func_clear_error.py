from tensorflow.python import pywrap_tfe
def clear_error(self):
    """Clears errors raised in this executor during execution."""
    pywrap_tfe.TFE_ExecutorClearError(self._handle)