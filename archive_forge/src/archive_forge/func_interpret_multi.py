import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
def interpret_multi(self, input, discourse_id=None, question=False, verbose=False):
    """
        Use Boxer to give a first order representation.

        :param input: list of str Input sentences to parse as a single discourse
        :param occur_index: bool Should predicates be occurrence indexed?
        :param discourse_id: str An identifier to be inserted to each occurrence-indexed predicate.
        :return: ``drt.DrtExpression``
        """
    discourse_ids = [discourse_id] if discourse_id is not None else None
    d, = self.interpret_multi_sents([input], discourse_ids, question, verbose)
    if not d:
        raise Exception(f'Unable to interpret: "{input}"')
    return d