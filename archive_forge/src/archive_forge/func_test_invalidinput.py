from wsme.exc import (ClientSideError, InvalidInput, MissingArgument,
def test_invalidinput():
    e = InvalidInput('field', 'badvalue', 'error message')
    assert e.faultstring == "Invalid input for field/attribute field. Value: 'badvalue'. error message", e.faultstring