from itertools import groupby
import numpy as np
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.u2 import U2Gate
from qiskit.circuit.library.standard_gates.u3 import U3Gate
from qiskit.circuit import ParameterExpression
from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.quaternion import Quaternion
from qiskit._accelerate.optimize_1q_gates import compose_u3_rust
@staticmethod
def yzy_to_zyz(xi, theta1, theta2, eps=1e-09):
    """Express a Y.Z.Y single qubit gate as a Z.Y.Z gate.

        Solve the equation

        .. math::

        Ry(theta1).Rz(xi).Ry(theta2) = Rz(phi).Ry(theta).Rz(lambda)

        for theta, phi, and lambda.

        Return a solution theta, phi, and lambda.
        """
    quaternion_yzy = Quaternion.from_euler([theta1, xi, theta2], 'yzy')
    euler = quaternion_yzy.to_zyz()
    quaternion_zyz = Quaternion.from_euler(euler, 'zyz')
    out_angles = (euler[1], euler[0], euler[2])
    abs_inner = abs(quaternion_zyz.data.dot(quaternion_yzy.data))
    if not np.allclose(abs_inner, 1, eps):
        raise TranspilerError('YZY and ZYZ angles do not give same rotation matrix.')
    out_angles = tuple((0 if np.abs(angle) < _CHOP_THRESHOLD else angle for angle in out_angles))
    return out_angles