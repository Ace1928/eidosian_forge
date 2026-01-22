from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder

        The loop can wake up just fine even if there are no timers in it.
        