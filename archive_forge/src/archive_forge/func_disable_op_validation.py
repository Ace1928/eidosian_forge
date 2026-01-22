import contextlib
@contextlib.contextmanager
def disable_op_validation(*, accept_debug_responsibility: bool=False):
    if not accept_debug_responsibility:
        raise ValueError('WARNING! Using disable_op_validation with invalid ops can cause mysterious and terrible things to happen. cirq-maintainers will not help you debug beyond this point!\nIf you still wish to continue, call this method with accept_debug_responsibility=True.')
    from cirq.ops import raw_types
    temp = raw_types._validate_qid_shape
    raw_types._validate_qid_shape = lambda *args: None
    try:
        yield None
    finally:
        raw_types._validate_qid_shape = temp