import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_with_no_getter(self):
    instance = ClassWithPropertyMissingGetter()
    try:
        instance.age += 1
    except Exception:
        self.fail('Having property with undefined getter/setter should not prevent the observed traits from being changed.')