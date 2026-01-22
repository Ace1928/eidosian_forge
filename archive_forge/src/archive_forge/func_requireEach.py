from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
def requireEach(someVersion: str) -> str:
    try:
        require_version('Gtk', someVersion)
    except ValueError as ve:
        return str(ve)
    else:
        return ''