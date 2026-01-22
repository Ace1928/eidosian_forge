import numpy as np
import pennylane as qml
Performs the `iterative quantum phase estimation <https://arxiv.org/pdf/quant-ph/0610214.pdf>`_ circuit.

    Given a unitary :math:`U`, this function applies the circuit for iterative quantum phase
    estimation and returns a list of mid-circuit measurements with qubit reset.

    Args:
      base (Operator): the phase estimation unitary, specified as an :class:`~.Operator`
      ancilla (Union[Wires, int, str]): the wire to be used for the estimation
      iters (int): the number of measurements to be performed

    Returns:
      list[MidMeasureMP]: the list of measurements performed

    .. seealso:: :class:`~.QuantumPhaseEstimation`, :func:`~.measure`

    **Example**

    .. code-block:: python

        dev = qml.device("default.qubit", shots=5)

        @qml.qnode(dev)
        def circuit():

          # Initial state
          qml.X(0)

          # Iterative QPE
          measurements = qml.iterative_qpe(qml.RZ(2.0, wires=[0]), ancilla=1, iters=3)

          return qml.sample(measurements)

    .. code-block:: pycon

        >>> print(circuit())
        [[0 0 1]
         [0 0 1]
         [0 0 1]
         [1 1 1]
         [0 0 1]]

    The output is an array of size ``(number of shots, number of iterations)``.

    .. code-block:: pycon

        >>> print(qml.draw(circuit, max_length=150)())

        0: ──X─╭RZ(2.00)⁴─────────────────╭RZ(2.00)²────────────────────────────╭RZ(2.00)¹────────────────────────────────────┤
        1: ──H─╰●──────────H──┤↗│  │0⟩──H─╰●──────────Rϕ(-1.57)──H──┤↗│  │0⟩──H─╰●──────────Rϕ(-1.57)──Rϕ(-0.79)──H──┤↗│  │0⟩─┤
                               ╚══════════════════════╩══════════════║══════════════════════║══════════╩══════════════║═══════╡ ╭Sample[MCM]
                                                                     ╚══════════════════════╩═════════════════════════║═══════╡ ├Sample[MCM]
                                                                                                                      ╚═══════╡ ╰Sample[MCM]
    