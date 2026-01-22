from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.core.expr import unchanged
from sympy.matrices import Matrix, SparseMatrix
from sympy.physics.quantum.commutator import Commutator as Comm
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.tensorproduct import TensorProduct as TP
from sympy.physics.quantum.tensorproduct import tensor_product_simp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qubit import Qubit, QubitBra
from sympy.physics.quantum.operator import OuterProduct
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
def test_issue_5923():
    assert TensorProduct(1, Qubit('1') * Qubit('1').dual) == TensorProduct(1, OuterProduct(Qubit(1), QubitBra(1)))