import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def prover9_input(self, goal, assumptions):
    """
        :see: Prover9Parent.prover9_input
        """
    s = 'clear(auto_denials).\n'
    return s + Prover9Parent.prover9_input(self, goal, assumptions)