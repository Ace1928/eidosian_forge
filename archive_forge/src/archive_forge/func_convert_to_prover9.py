import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def convert_to_prover9(input):
    """
    Convert a ``logic.Expression`` to Prover9 format.
    """
    if isinstance(input, list):
        result = []
        for s in input:
            try:
                result.append(_convert_to_prover9(s.simplify()))
            except:
                print('input %s cannot be converted to Prover9 input syntax' % input)
                raise
        return result
    else:
        try:
            return _convert_to_prover9(input.simplify())
        except:
            print('input %s cannot be converted to Prover9 input syntax' % input)
            raise