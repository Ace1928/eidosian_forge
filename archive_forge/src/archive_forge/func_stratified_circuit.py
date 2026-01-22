import itertools
from typing import TYPE_CHECKING, Type, Callable, Dict, Optional, Union, Iterable, Sequence, List
from cirq import ops, circuits, protocols, _import
from cirq.transformers import transformer_api
@transformer_api.transformer(add_deep_support=True)
def stratified_circuit(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None, categories: Iterable[Category]=()) -> 'cirq.Circuit':
    """Repacks avoiding simultaneous operations with different classes.

    This transforms the given circuit to ensure that no operations of different categories are
    found in the same moment. Makes no optimality guarantees.
    Tagged Operations marked with any of `context.tags_to_ignore` will be treated as a separate
    category will be left in their original moments without stratification.

    Args:
        circuit: The circuit whose operations should be re-arranged. Will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        categories: A list of classifiers picking out certain operations. There are several ways
            to specify a classifier. You can pass in a gate instance (e.g. `cirq.X`),
            a gate type (e.g. `cirq.XPowGate`), an operation instance
            (e.g. `cirq.X(cirq.LineQubit(0))`), an operation type (e.g.`cirq.CircuitOperation`),
            or an arbitrary operation predicate (e.g. `lambda op: len(op.qubits) == 2`).

    Returns:
        A copy of the original circuit, but with re-arranged operations.
    """
    classifiers = _get_classifiers(circuit, categories)
    smallest_depth = protocols.num_qubits(circuit) * len(circuit) + 1
    shortest_stratified_circuit = circuits.Circuit()
    reversed_circuit = circuit[::-1]
    for ordered_classifiers in itertools.permutations(classifiers):
        solution = _stratify_circuit(circuit, classifiers=ordered_classifiers, context=context or transformer_api.TransformerContext())
        if len(solution) < smallest_depth:
            shortest_stratified_circuit = solution
            smallest_depth = len(solution)
        solution = _stratify_circuit(reversed_circuit, classifiers=ordered_classifiers, context=context or transformer_api.TransformerContext())[::-1]
        if len(solution) < smallest_depth:
            shortest_stratified_circuit = solution
            smallest_depth = len(solution)
    return shortest_stratified_circuit