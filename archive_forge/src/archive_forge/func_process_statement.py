from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
def process_statement(self, execute_observed):
    for rule in self.rules:
        rule.process_statement(execute_observed)
        if rule.is_consumed:
            self.is_consumed = True
            break
    else:
        self.errormessage = list(self.rules)[0].errormessage