from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
@property
def rules_beta(self):
    return self.split_alpha_beta()[1]