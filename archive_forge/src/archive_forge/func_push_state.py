import re
import sys
import types
import copy
import os
import inspect
def push_state(self, state):
    self.lexstatestack.append(self.lexstate)
    self.begin(state)