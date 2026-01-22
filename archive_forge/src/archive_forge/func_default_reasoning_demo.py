from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def default_reasoning_demo():
    lexpr = Expression.fromstring
    premises = []
    premises.append(lexpr('all x.(elephant(x)        -> animal(x))'))
    premises.append(lexpr('all x.(bird(x)            -> animal(x))'))
    premises.append(lexpr('all x.(dove(x)            -> bird(x))'))
    premises.append(lexpr('all x.(ostrich(x)         -> bird(x))'))
    premises.append(lexpr('all x.(flying_ostrich(x)  -> ostrich(x))'))
    premises.append(lexpr('all x.((animal(x)  & -Ab1(x)) -> -fly(x))'))
    premises.append(lexpr('all x.((bird(x)    & -Ab2(x)) -> fly(x))'))
    premises.append(lexpr('all x.((ostrich(x) & -Ab3(x)) -> -fly(x))'))
    premises.append(lexpr('all x.(bird(x)           -> Ab1(x))'))
    premises.append(lexpr('all x.(ostrich(x)        -> Ab2(x))'))
    premises.append(lexpr('all x.(flying_ostrich(x) -> Ab3(x))'))
    premises.append(lexpr('elephant(E)'))
    premises.append(lexpr('dove(D)'))
    premises.append(lexpr('ostrich(O)'))
    prover = Prover9Command(None, premises)
    command = UniqueNamesProver(ClosedWorldProver(prover))
    for a in command.assumptions():
        print(a)
    print_proof('-fly(E)', premises)
    print_proof('fly(D)', premises)
    print_proof('-fly(O)', premises)