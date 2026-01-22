import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
@on_trait_change('discounted')
def discounted_updated(self, event):
    self.discounted_events.append(event)