from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def qubit(self, arg, init=None):
    self.labels.append(arg)
    if init:
        self.inits[arg] = init