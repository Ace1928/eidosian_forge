from tensorflow.python import pywrap_tfe
def pop_tape(tape):
    """Pops the given tape in the stack."""
    pywrap_tfe.TFE_Py_TapeSetRemove(tape._tape)