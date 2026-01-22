from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def toffoli(self, a1, a2, a3):
    i1, i2, i3 = self.indices([a1, a2, a3])
    self.circuit.append(CGateS((i1, i2), X(i3)))