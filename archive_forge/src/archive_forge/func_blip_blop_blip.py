import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
@updating.updated_kwarg_default_value('type', 'cat', 'feline')
def blip_blop_blip(type='cat'):
    return 'The %s meowed quietly' % type