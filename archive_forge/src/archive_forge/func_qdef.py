from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def qdef(self, name, ncontrols, symbol):
    from sympy.physics.quantum.circuitplot import CreateOneQubitGate, CreateCGate
    ncontrols = int(ncontrols)
    command = fixcommand(name)
    symbol = stripquotes(symbol)
    if ncontrols > 0:
        self.defs[command] = CreateCGate(symbol)
    else:
        self.defs[command] = CreateOneQubitGate(symbol)