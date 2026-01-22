import random
import threading
import time
import unittest
from traits.api import Enum, HasStrictTraits
from traits.util.async_trait_wait import wait_for_condition
def make_random_changes(self, change_count):
    for _ in range(change_count):
        time.sleep(random.uniform(0.1, 0.3))
        self.colour = self._next_colour[self.colour]