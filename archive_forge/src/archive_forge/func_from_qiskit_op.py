from collections import defaultdict
from importlib import metadata
from sys import version_info
def from_qiskit_op(qiskit_op, params=None, wires=None):
    """Converts a Qiskit `SparsePauliOp <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp>`__
    into a PennyLane :class:`Operator <pennylane.operation.Operator>`.

    .. note::

        This function depends upon the PennyLane-Qiskit plugin. Follow the
        `installation instructions <https://docs.pennylane.ai/projects/qiskit/en/latest/installation.html>`__
        to get up and running. You may need to restart your kernel if you are running in a notebook
        environment.

    Args:
        qiskit_op (qiskit.quantum_info.SparsePauliOp): a ``SparsePauliOp`` created in Qiskit
        params (Any): optional assignment of coefficient values for the ``SparsePauliOp``; see the
            `Qiskit documentation <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp#assign_parameters>`_
            to learn more about the expected format of these parameters
        wires (Sequence | None): optional assignment of wires for the converted ``SparsePauliOp``;
            if the original ``SparsePauliOp`` acted on :math:`N` qubits, then this must be a
            sequence of length :math:`N`

    Returns:
        Operator: The PennyLane operator, created based on the input Qiskit
        ``SparsePauliOp`` object.

    .. note::

        The wire ordering convention differs between PennyLane and Qiskit: PennyLane wires are
        enumerated from left to right, while the Qiskit convention is to enumerate from right to
        left. This means a ``SparsePauliOp`` term defined by the string ``"XYZ"`` applies ``Z`` on
        wire 0, ``Y`` on wire 1, and ``X`` on wire 2. For more details, see the
        `String representation <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Pauli>`_
        section of the Qiskit documentation for the ``Pauli`` class.

    **Example**

    Consider the following script which creates a Qiskit ``SparsePauliOp``:

    .. code-block:: python

        from qiskit.quantum_info import SparsePauliOp

        qiskit_op = SparsePauliOp(["II", "XY"])

    The ``SparsePauliOp`` contains two terms and acts over two qubits:

    >>> qiskit_op
    SparsePauliOp(['II', 'XY'],
                  coeffs=[1.+0.j, 1.+0.j])

    To convert the ``SparsePauliOp`` into a PennyLane :class:`pennylane.operation.Operator`, use:

    >>> import pennylane as qml
    >>> qml.from_qiskit_op(qiskit_op)
    I(0) + X(1) @ Y(0)

    .. details::
        :title: Usage Details

        You can convert a parameterized ``SparsePauliOp`` into a PennyLane operator by assigning
        literal values to each coefficient parameter. For example, the script

        .. code-block:: python

            import numpy as np
            from qiskit.circuit import Parameter

            a, b, c = [Parameter(var) for var in "abc"]
            param_qiskit_op = SparsePauliOp(["II", "XZ", "YX"], coeffs=np.array([a, b, c]))

        defines a ``SparsePauliOp`` with three coefficients (parameters):

        >>> param_qiskit_op
        SparsePauliOp(['II', 'XZ', 'YX'],
              coeffs=[ParameterExpression(1.0*a), ParameterExpression(1.0*b),
         ParameterExpression(1.0*c)])

        The ``SparsePauliOp`` can be converted into a PennyLane operator by calling the conversion
        function and specifying the value of each parameter using the ``params`` argument:

        >>> qml.from_qiskit_op(param_qiskit_op, params={a: 2, b: 3, c: 4})
        (
            (2+0j) * I(0)
          + (3+0j) * (X(1) @ Z(0))
          + (4+0j) * (Y(1) @ X(0))
        )

        Similarly, a custom wire mapping can be applied to a ``SparsePauliOp`` as follows:

        >>> wired_qiskit_op = SparsePauliOp("XYZ")
        >>> wired_qiskit_op
        SparsePauliOp(['XYZ'],
              coeffs=[1.+0.j])
        >>> qml.from_qiskit_op(wired_qiskit_op, wires=[3, 5, 7])
        Y(5) @ Z(3) @ X(7)
    """
    try:
        return load(qiskit_op, format='qiskit_op', params=params, wires=wires)
    except ValueError as e:
        if e.args[0].split('.')[0] == 'Converter does not exist':
            raise RuntimeError(_MISSING_QISKIT_PLUGIN_MESSAGE) from e
        raise e