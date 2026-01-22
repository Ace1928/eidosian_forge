import doctest
import os
from breezy import tests
def make_new_test_id(test):
    new_id = '{}.DocFileTest({})'.format(__name__, test.id())
    return lambda: new_id