import unittest
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
def bar_to_foo_adapter(bar):
    return Foo()