from tensorflow.python import pywrap_tfe
@property
def is_cancelled(self):
    """Returns `True` if `CancellationManager.start_cancel` has been called."""
    return pywrap_tfe.TFE_CancellationManagerIsCancelled(self._impl)