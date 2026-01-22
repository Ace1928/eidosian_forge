import pytest
import cirq
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_best_of_gets_longest_needs_minimum():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    assert GridQubitLineTuple.best_of([[]], 0) == ()
    assert GridQubitLineTuple.best_of([[], [q00]], 0) == ()
    assert GridQubitLineTuple.best_of([[q00], []], 0) == ()
    assert GridQubitLineTuple.best_of([[], [q00]], 1) == (q00,)
    assert GridQubitLineTuple.best_of([[q00], []], 1) == (q00,)
    assert GridQubitLineTuple.best_of([[q00, q01], [q00]], 1) == (q00,)
    assert GridQubitLineTuple.best_of([[q00, q01], [q00]], 2) == (q00, q01)
    assert GridQubitLineTuple.best_of([[q00, q01]], 2) == (q00, q01)
    assert GridQubitLineTuple.best_of([], 0) == ()
    with pytest.raises(NotFoundError):
        _ = GridQubitLineTuple.best_of([[]], 1)
    with pytest.raises(NotFoundError):
        _ = GridQubitLineTuple.best_of([[q00]], 2)