import itertools
from qiskit.transpiler import CouplingMap, Target, AnalysisPass, TranspilerError
from qiskit.transpiler.passes.layout.vf2_layout import VF2Layout
from qiskit._accelerate.error_map import ErrorMap
Minimizes the set of extra edges involved in the layout. This iteratively
        removes extra edges from the coupling map and uses VF2 to check if a layout
        still exists. This is reasonably efficiently as it only looks for a local
        minimum.
        