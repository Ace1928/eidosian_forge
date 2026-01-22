from typing import Any, Dict, List, Sequence, Type, Union
import cirq
List of transformers which should be run after decomposing individual operations.

        If `eject_paulis` is enabled in the constructor, adds `cirq.eject_phased_paulis` and
        `cirq.eject_z` in addition to postprocess_transformers already available in
        `cirq.CompilationTargetGateset`.
        