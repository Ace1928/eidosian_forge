import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_state_attribute(self):
    from pecan.secure import Any, Protected
    assert repr(Any) == '<SecureState Any>'
    assert bool(Any) is False
    assert repr(Protected) == '<SecureState Protected>'
    assert bool(Protected) is True