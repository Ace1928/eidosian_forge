import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def phone_reaction(old_state, new_state, event, chat_iter):
    try:
        next(chat_iter)
    except StopIteration:
        return 'finish'
    else:
        return 'chat'