import abc
import dataclasses
from typing import Iterable, List, Optional
import cirq
from cirq.protocols.circuit_diagram_info_protocol import CircuitDiagramInfoArgs
def resolve_operation(operation: cirq.Operation, resolvers: Iterable[SymbolResolver]) -> SymbolInfo:
    """Builds a SymbolInfo object based off of a designated operation
    and list of resolvers. The latest resolver takes precendent.

    Args:
        operation: the cirq.Operation object to resolve
        resolvers: a list of SymbolResolvers which provides instructions
        on how to build SymbolInfo objects.

    Raises:
        ValueError: if the operation cannot be resolved into a symbol.
    """
    symbol_info = None
    for resolver in resolvers:
        info = resolver(operation)
        if info is not None:
            symbol_info = info
    if symbol_info is None:
        raise ValueError(f'Cannot resolve operation: {operation}')
    return symbol_info