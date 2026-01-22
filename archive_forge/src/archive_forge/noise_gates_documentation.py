from pyquil.quilbase import Gate
from pyquil.quilatom import Parameter, unpack_qubit
from pyquil.external.rpcq import CompilerISA, Supported1QGate, Supported2QGate, GateInfo, Edge
from typing import List, Optional
import logging

    Generate the gate set associated with an ISA for which QVM noise is supported.

    :param isa: The instruction set architecture for a QPU.
    :return: A list of Gate objects encapsulating all gates compatible with the ISA.
    