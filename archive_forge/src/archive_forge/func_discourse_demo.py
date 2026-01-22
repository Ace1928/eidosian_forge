import os
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import add, and_
from nltk.data import show_cfg
from nltk.inference.mace import MaceCommand
from nltk.inference.prover9 import Prover9Command
from nltk.parse import load_parser
from nltk.parse.malt import MaltParser
from nltk.sem.drt import AnaphoraResolutionException, resolve_anaphora
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Expression
from nltk.tag import RegexpTagger
def discourse_demo(reading_command=None):
    """
    Illustrate the various methods of ``DiscourseTester``
    """
    dt = DiscourseTester(['A boxer walks', 'Every boxer chases a girl'], reading_command)
    dt.models()
    print()
    print()
    dt.sentences()
    print()
    dt.readings()
    print()
    dt.readings(threaded=True)
    print()
    dt.models('d1')
    dt.add_sentence('John is a boxer')
    print()
    dt.sentences()
    print()
    dt.readings(threaded=True)
    print()
    dt = DiscourseTester(['A student dances', 'Every student is a person'], reading_command)
    print()
    dt.add_sentence('No person dances', consistchk=True)
    print()
    dt.readings()
    print()
    dt.retract_sentence('No person dances', verbose=True)
    print()
    dt.models()
    print()
    dt.readings('A person dances')
    print()
    dt.add_sentence('A person dances', informchk=True)
    dt = DiscourseTester(['Vincent is a boxer', 'Fido is a boxer', 'Vincent is married', 'Fido barks'], reading_command)
    dt.readings(filter=True)
    import nltk.data
    background_file = os.path.join('grammars', 'book_grammars', 'background.fol')
    background = nltk.data.load(background_file)
    print()
    dt.add_background(background, verbose=False)
    dt.background()
    print()
    dt.readings(filter=True)
    print()
    dt.models()