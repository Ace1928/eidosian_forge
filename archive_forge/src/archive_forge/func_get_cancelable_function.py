from tensorflow.python import pywrap_tfe
def get_cancelable_function(self, concrete_function):

    def cancellable(*args, **kwargs):
        with CancellationManagerContext(self):
            return concrete_function(*args, **kwargs)
    return cancellable