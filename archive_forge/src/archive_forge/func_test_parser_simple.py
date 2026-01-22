import pytest
from jeepney.low_level import *
def test_parser_simple():
    msg = Parser().feed(HELLO_METHOD_CALL)[0]
    assert msg.header.fields[HeaderFields.member] == 'Hello'