import random
import threading
import time
import unittest
from traits.api import Enum, HasStrictTraits
from traits.util.async_trait_wait import wait_for_condition
def test_wait_for_condition_failure(self):
    lights = TrafficLights(colour='Green')
    t = threading.Thread(target=lights.make_random_changes, args=(2,))
    t.start()
    self.assertRaises(RuntimeError, wait_for_condition, condition=lambda l: l.colour == 'RedAndAmber', obj=lights, trait='colour', timeout=5.0)
    t.join()