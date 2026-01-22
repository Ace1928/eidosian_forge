from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.tree_util import register_pytree_node_class
import pennylane as qml
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian
@staticmethod
def from_hamiltonian(H: ParametrizedHamiltonian, *, dense: bool=False, wire_order=None):
    """Convert a ``ParametrizedHamiltonian`` into a jax pytree object.

        Args:
            H (ParametrizedHamiltonian): parametrized Hamiltonian to convert
            dense (bool, optional): Decide wether a dense/sparse matrix is used. Defaults to False.
            wire_order (list, optional): Wire order of the returned ``JaxParametrizedOperator``.
                Defaults to None.

        Returns:
            ParametrizedHamiltonianPytree: pytree object
        """
    make_array = jnp.array if dense else sparse.BCSR.fromdense
    if len(H.ops_fixed) > 0:
        mat_fixed = make_array(qml.matrix(H.H_fixed(), wire_order=wire_order))
    else:
        mat_fixed = None
    mats_parametrized = tuple((make_array(qml.matrix(op, wire_order=wire_order)) for op in H.ops_parametrized))
    if isinstance(H, HardwareHamiltonian):
        return ParametrizedHamiltonianPytree(mat_fixed, mats_parametrized, H.coeffs_parametrized, reorder_fn=H.reorder_fn)
    return ParametrizedHamiltonianPytree(mat_fixed, mats_parametrized, H.coeffs_parametrized, reorder_fn=None)