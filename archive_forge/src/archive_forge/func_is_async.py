from tensorflow.python import pywrap_tfe
def is_async(self):
    return pywrap_tfe.TFE_ExecutorIsAsync(self._handle)