import re
from typing import Tuple
import pytest
from .._receivebuffer import ReceiveBuffer
@pytest.mark.parametrize('data', [pytest.param((b'HTTP/1.1 200 OK\r\n', b'Content-type: text/plain\r\n', b'Connection: close\r\n', b'\r\n', b'Some body'), id='with_crlf_delimiter'), pytest.param((b'HTTP/1.1 200 OK\n', b'Content-type: text/plain\n', b'Connection: close\n', b'\n', b'Some body'), id='with_lf_only_delimiter'), pytest.param((b'HTTP/1.1 200 OK\n', b'Content-type: text/plain\r\n', b'Connection: close\n', b'\n', b'Some body'), id='with_mixed_crlf_and_lf')])
def test_receivebuffer_for_invalid_delimiter(data: Tuple[bytes]) -> None:
    b = ReceiveBuffer()
    for line in data:
        b += line
    lines = b.maybe_extract_lines()
    assert lines == [b'HTTP/1.1 200 OK', b'Content-type: text/plain', b'Connection: close']
    assert bytes(b) == b'Some body'