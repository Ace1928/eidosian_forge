import unittest
from twisted.internet import defer
This test creates an unhandled Deferred and leaves it in a cycle.

    The Deferred is left in a cycle so that the garbage collector won't pick it
    up immediately.  We were having some problems where unhandled Deferreds in
    one test were failing random other tests. (See #1507, #1213)
    