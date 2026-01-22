from __future__ import annotations
import email.message
import email.parser
from io import BytesIO, StringIO
from typing import IO, AnyStr, Callable
from twisted.mail import bounce
from twisted.trial import unittest
def test_bounceBigMessage(self) -> None:
    """
        L{twisted.mail.bounce.generateBounce} with big L{unicode} and
        L{bytes} messages.
        """
    header = b'From: Moshe Zadka <moshez@example.com>\nTo: nonexistent@example.org\nSubject: test\n\n'
    self._bounceBigMessage(header, b'Test test\n' * 10000, BytesIO)
    self._bounceBigMessage(header.decode('utf-8'), 'More test\n' * 10000, StringIO)