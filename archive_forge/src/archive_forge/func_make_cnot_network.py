import logging
import numpy as np
def make_cnot_network(num_qubits: int, network_layout: str='spin', connectivity_type: str='full', depth: int=0) -> np.ndarray:
    """
    Generates a network consisting of building blocks each containing a CNOT gate and possibly some
    single-qubit ones. This network models a quantum operator in question. Note, each building
    block has 2 input and outputs corresponding to a pair of qubits. What we actually return here
    is a chain of indices of qubit pairs shared by every building block in a row.

    Args:
        num_qubits: number of qubits.
        network_layout: type of network geometry, ``{"sequ", "spin", "cart", "cyclic_spin",
            "cyclic_line"}``.
        connectivity_type: type of inter-qubit connectivity, ``{"full", "line", "star"}``.
        depth: depth of the CNOT-network, i.e. the number of layers, where each layer consists of
            a single CNOT-block; default value will be selected, if ``L <= 0``.

    Returns:
        A matrix of size ``(2, N)`` matrix that defines layers in cnot-network, where ``N``
            is either equal ``L``, or defined by a concrete type of the network.

    Raises:
         ValueError: if unsupported type of CNOT-network layout or number of qubits or combination
            of parameters are passed.
    """
    if num_qubits < 2:
        raise ValueError('Number of qubits must be greater or equal to 2')
    if depth <= 0:
        new_depth = _lower_limit(num_qubits)
        logger.debug('Number of CNOT units chosen as the lower limit: %d, got a non-positive value: %d', new_depth, depth)
        depth = new_depth
    if network_layout == 'sequ':
        links = _get_connectivity(num_qubits=num_qubits, connectivity=connectivity_type)
        return _sequential_network(num_qubits=num_qubits, links=links, depth=depth)
    elif network_layout == 'spin':
        return _spin_network(num_qubits=num_qubits, depth=depth)
    elif network_layout == 'cart':
        cnots = _cartan_network(num_qubits=num_qubits)
        logger.debug('Optimal lower bound: %d; Cartan CNOTs: %d', _lower_limit(num_qubits), cnots.shape[1])
        return cnots
    elif network_layout == 'cyclic_spin':
        if connectivity_type != 'full':
            raise ValueError(f"'{network_layout}' layout expects 'full' connectivity")
        return _cyclic_spin_network(num_qubits, depth)
    elif network_layout == 'cyclic_line':
        if connectivity_type != 'line':
            raise ValueError(f"'{network_layout}' layout expects 'line' connectivity")
        return _cyclic_line_network(num_qubits, depth)
    else:
        raise ValueError(f'Unknown type of CNOT-network layout, expects one of {_NETWORK_LAYOUTS}, got {network_layout}')