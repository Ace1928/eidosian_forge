import re
import sys
import types
import copy
import os
import inspect
def pop_state(self):
    self.begin(self.lexstatestack.pop())