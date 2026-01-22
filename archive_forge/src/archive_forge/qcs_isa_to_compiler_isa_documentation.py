from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict

    Signals an error when creating a ``CompilerISA`` due to the operators
    in the QCS ``InstructionSetArchitecture``. This may raise as a consequence
    of unsupported gates as well as missing nodes or edges.
    