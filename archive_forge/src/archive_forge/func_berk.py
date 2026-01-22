import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
@property
@moves.moved_property('bark', category=PendingDeprecationWarning)
def berk(self):
    return self.bark