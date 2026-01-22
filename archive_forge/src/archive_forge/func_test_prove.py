import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def test_prove(arguments):
    """
    Try some proofs and exhibit the results.
    """
    for goal, assumptions in arguments:
        g = Expression.fromstring(goal)
        alist = [Expression.fromstring(a) for a in assumptions]
        p = Prover9Command(g, assumptions=alist).prove()
        for a in alist:
            print('   %s' % a)
        print(f'|- {g}: {p}\n')