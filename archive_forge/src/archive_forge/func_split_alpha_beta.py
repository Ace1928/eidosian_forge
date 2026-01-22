from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def split_alpha_beta(self):
    """split proved rules into alpha and beta chains"""
    rules_alpha = []
    rules_beta = []
    for a, b in self.proved_rules:
        if isinstance(a, And):
            rules_beta.append((a, b))
        else:
            rules_alpha.append((a, b))
    return (rules_alpha, rules_beta)