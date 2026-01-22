from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport
def makeUI(self):
    return DummyUI()