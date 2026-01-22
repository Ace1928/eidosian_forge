import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def init_with(nested_machine):
    queried_for.append(nested_machine)
    return None