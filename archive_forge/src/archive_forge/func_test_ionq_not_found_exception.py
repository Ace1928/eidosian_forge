import cirq_ionq as ionq
def test_ionq_not_found_exception():
    ex = ionq.IonQNotFoundException(message='Where are you')
    assert str(ex) == "Status code: 404, Message: 'Where are you'"
    assert ex.status_code == 404