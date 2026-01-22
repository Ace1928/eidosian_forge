import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
 Test traits class with static and dynamic listeners.

    Changing `baz` triggers a dynamic listeners that modifies `bar`, which
    triggers one dynamic and one static listeners.
    