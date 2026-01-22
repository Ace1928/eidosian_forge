import cirq_ionq as ionq
def test_ionq_exception():
    ex = ionq.IonQException(message='Hello', status_code=500)
    assert str(ex) == "Status code: 500, Message: 'Hello'"
    assert ex.status_code == 500