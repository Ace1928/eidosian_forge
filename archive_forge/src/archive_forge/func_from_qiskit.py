from collections import defaultdict
from importlib import metadata
from sys import version_info
def from_qiskit(quantum_circuit, measurements=None):
    """Converts a Qiskit `QuantumCircuit <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit>`_
    into a PennyLane :ref:`quantum function <intro_vcirc_qfunc>`.

    .. note::

        This function depends upon the PennyLane-Qiskit plugin. Follow the
        `installation instructions <https://docs.pennylane.ai/projects/qiskit/en/latest/installation.html>`__
        to get up and running. You may need to restart your kernel if you are running in a notebook
        environment.

    Args:
        quantum_circuit (qiskit.QuantumCircuit): a quantum circuit created in Qiskit
        measurements (None | MeasurementProcess | list[MeasurementProcess]): an optional PennyLane
            measurement or list of PennyLane measurements that overrides any terminal measurements
            that may be present in the input circuit

    Returns:
        function: The PennyLane quantum function, created based on the input Qiskit
        ``QuantumCircuit`` object.

    **Example:**

    .. code-block:: python

        import pennylane as qml
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.rx(0.785, 0)
        qc.ry(1.57, 1)

        my_qfunc = qml.from_qiskit(qc)

    The ``my_qfunc`` function can now be used within QNodes, as a two-wire quantum
    template. We can also pass ``wires`` when calling the returned template to define
    which wires it should operate on. If no wires are passed, it will default
    to sequential wire labels starting at 0.

    .. code-block:: python

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            my_qfunc(wires=["a", "b"])
            return qml.expval(qml.Z("a")), qml.var(qml.Z("b"))

    >>> circuit()
    (tensor(0.70738827, requires_grad=True),
    tensor(0.99999937, requires_grad=True))

    The measurements can also be passed directly to the function when creating the
    quantum function, making it possible to create a PennyLane circuit with
    :class:`qml.QNode <pennylane.QNode>`:

    >>> measurements = [qml.expval(qml.Z(0)), qml.var(qml.Z(1))]
    >>> circuit = qml.QNode(qml.from_qiskit(qc, measurements), dev)
    >>> circuit()
    (tensor(0.70738827, requires_grad=True),
    tensor(0.99999937, requires_grad=True))

    .. note::

        The ``measurements`` keyword allows one to add a list of PennyLane measurements
        that will **override** any terminal measurements present in the ``QuantumCircuit``,
        so that they are not performed before the operations specified in ``measurements``.
        ``measurements=None``.

    If an existing ``QuantumCircuit`` already contains measurements, ``from_qiskit``
    will return those measurements, provided that they are not overriden as shown above.
    These measurements can be used, e.g., for conditioning with
    :func:`qml.cond() <~.cond>`, or simply included directly within the QNode's return:

    .. code-block:: python

       qc = QuantumCircuit(2, 2)
       qc.rx(np.pi, 0)
       qc.measure_all()

       @qml.qnode(dev)
       def circuit():
           # Since measurements=None, the measurements present in the QuantumCircuit are returned.
           measurements = qml.from_qiskit(qc)()
           return [qml.expval(m) for m in measurements]

    >>> circuit()
    [tensor(1., requires_grad=True), tensor(0., requires_grad=True)]

    .. note::

        The ``measurements`` returned from a ``QuantumCircuit`` are in the computational basis
        with 0 corresponding to :math:`|0\\rangle` and 1 corresponding to :math:`|1 \\rangle`. This
        corresponds to the :math:`|1 \\rangle \\langle 1|` observable rather than the :math:`Z` Pauli
        operator.

    See below for more information regarding how to translate more complex circuits from Qiskit to
    PennyLane, including handling parameterized Qiskit circuits, mid-circuit measurements, and
    classical control flows.

    .. details::
        :title: Parameterized Quantum Circuits

        A Qiskit ``QuantumCircuit`` is parameterized if it contains
        `Parameter <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Parameter>`__ or
        `ParameterVector <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.ParameterVector>`__
        references that need to be given defined values to evaluate the circuit. These can be passed
        to the generated quantum function as keyword or positional arguments. If we define a
        parameterized circuit:

        .. code-block:: python

            from qiskit.circuit import QuantumCircuit, Parameter

            angle0 = Parameter("x")
            angle1 = Parameter("y")

            qc = QuantumCircuit(2, 2)
            qc.rx(angle0, 0)
            qc.ry(angle1, 1)
            qc.cx(1, 0)

        Then this circuit can be converted into a differentiable circuit in PennyLane and
        executed:

        .. code-block:: python

            import pennylane as qml
            from pennylane import numpy as np

            dev = qml.device("default.qubit")

            qfunc = qml.from_qiskit(qc, measurements=qml.expval(qml.Z(0)))
            circuit = qml.QNode(qfunc, dev)

        Now, ``circuit`` has a signature of ``(x, y)``. The parameters are ordered alphabetically.

        >>> x = np.pi / 4
        >>> y = 0
        >>> circuit(x, y)
        tensor(0.70710678, requires_grad=True)

        >>> qml.grad(circuit, argnum=[0, 1])(np.pi/4, np.pi/6)
        (array(-0.61237244), array(-0.35355339))

        The ``QuantumCircuit`` may also be parameterized with a ``ParameterVector``. These can be
        similarly converted:

        .. code-block:: python

            from qiskit.circuit import ParameterVector

            angles = ParameterVector("angles", 2)

            qc = QuantumCircuit(2, 2)
            qc.rx(angles[0], 0)
            qc.ry(angles[1], 1)
            qc.cx(1, 0)

            @qml.qnode(dev)
            def circuit(angles):
                qml.from_qiskit(qc)(angles)
                return qml.expval(qml.Z(0))

        >>> angles = [3.1, 0.45]
        >>> circuit(angles)
        tensor(-0.89966835, requires_grad=True)


    .. details::
        :title: Measurements and Classical Control Flows

        When ``measurement=None``, all of the measurements performed in the ``QuantumCircuit`` will
        be returned by the quantum function in the form of a :ref:`mid-circuit measurement
        <mid_circuit_measurements>`. For example, if we define a ``QuantumCircuit`` with
        measurements:

        .. code-block:: python

            import pennylane as qml
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.measure(0, 0)
            qc.rz(0.24, [0])
            qc.cx(0, 1)
            qc.measure_all()

        Then we can create a PennyLane circuit that uses this as a sub-circuit, and performs
        additional operations conditional on the results. We can also calculate standard mid-circuit
        measurement statistics, like expectation value, on the returned measurements:

        .. code-block:: python

            @qml.qnode(qml.device("default.qubit"))
            def circuit():
                # apply the QuantumCircuit and retrieve the measurements
                mid_measure0, m0, m1 = qml.from_qiskit(qc)()

                # conditionally apply an additional operation based on the results
                qml.cond(mid_measure0==0, qml.RX)(np.pi/2, 0)

                # return the expectation value of one of the mid-circuit measurements, and a terminal measurement
                return qml.expval(mid_measure0), qml.expval(m1)

        >>> circuit()
        (tensor(0.5, requires_grad=True), tensor(0.5, requires_grad=True))

        .. note::

            The order of mid-circuit measurements returned by `qml.from_qiskit()` in the example
            above is determined by the order in which measurements appear in the input Qiskit
            ``QuantumCircuit``.

        Furthermore, the Qiskit `IfElseOp <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.IfElseOp>`__,
        `SwitchCaseOp <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.SwitchCaseOp>`__ and
        `c_if <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Instruction#c_if>`__
        conditional workflows are automatically translated into their PennyLane counterparts during
        conversion. For example, if we construct a ``QuantumCircuit`` with these workflows:

        .. code-block:: python

            qc = QuantumCircuit(4, 1)
            qc.h(0)
            qc.measure(0, 0)

            # Use an `IfElseOp` operation.
            noop = QuantumCircuit(1)
            flip_x = QuantumCircuit(1)
            flip_x.x(0)
            qc.if_else((qc.clbits[0], True), flip_x, noop, [1], [])

            # Use a `SwitchCaseOp` operation.
            with qc.switch(qc.clbits[0]) as case:
                with case(0):
                    qc.y(2)

            # Use the `c_if()` function.
            qc.z(3).c_if(qc.clbits[0], True)

            qc.measure_all()

        We can convert the ``QuantumCircuit`` into a PennyLane quantum function using:

        .. code-block:: python

            dev = qml.device("default.qubit")

            measurements = [qml.expval(qml.Z(i)) for i in range(qc.num_qubits)]
            cond_circuit = qml.QNode(qml.from_qiskit(qc, measurements=measurements), dev)

        The result is:

        >>> print(qml.draw(cond_circuit)())
        0: ──H──┤↗├──────────╭||─┤  <Z>
        1: ──────║───X───────├||─┤  <Z>
        2: ──────║───║──Y────├||─┤  <Z>
        3: ──────║───║──║──Z─╰||─┤  <Z>
                 ╚═══╩══╩══╝
    """
    try:
        return load(quantum_circuit, format='qiskit', measurements=measurements)
    except ValueError as e:
        if e.args[0].split('.')[0] == 'Converter does not exist':
            raise RuntimeError(_MISSING_QISKIT_PLUGIN_MESSAGE) from e
        raise e