from dataclasses import dataclass
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian, HardwarePulse, drive
from pennylane.wires import Wires
from pennylane.pulse.hardware_hamiltonian import _reorder_parameters
def rydberg_drive(amplitude, phase, detuning, wires):
    """Returns a :class:`ParametrizedHamiltonian` representing the action of a driving laser field

    .. math::

        \\frac{1}{2} \\Omega(t) \\sum_{q \\in \\text{wires}} (\\cos(\\phi(t))\\sigma_q^x - \\sin(\\phi(t))\\sigma_q^y) -
        \\delta(t) \\sum_{q \\in \\text{wires}} n_q

    where :math:`\\Omega/(2\\pi)`, :math:`\\phi` and :math:`\\delta/(2\\pi)` correspond to the amplitude, phase,
    and detuning of the laser, :math:`q` corresponds to the wire index, and
    :math:`\\sigma_q^\\alpha` for :math:`\\alpha = x,y` are the Pauli matrices on the corresponding
    qubit. Finally, :math:`n_q=\\frac{1}{2}(\\mathbb{I}_q-\\sigma_q^z)` is the number operator on qubit :math:`q`.

    .. note::
        For hardware execution, input time is expected to be in units of :math:`\\mu\\text{s}`, and the frequency
        in units of MHz. It is recommended to also follow this convention for simulation,
        as it avoids numerical problems due to using very large and very small numbers. Frequency inputs will be
        converted internally to angular frequency, such that ``amplitude`` :math:`= \\Omega(t)/ (2 \\pi)` and
        ``detuning`` :math:`= \\delta(t) / (2 \\pi)`.

    This driving term can be combined with an interaction term to create a Hamiltonian describing a
    driven Rydberg atom system. Multiple driving terms can be combined by summing them (see example).

    Args:
        amplitude (Union[float, Callable]): Float or callable representing the amplitude of a laser field.
            This should be in units of frequency (MHz), and will be converted to amplitude in angular frequency,
            :math:`\\Omega(t)`, internally where needed, i.e. multiplied by :math:`2 \\pi`.
        phase (Union[float, Callable]): float or callable representing the phase (in radians) of the laser field
        detuning (Union[float, Callable]): Float or callable representing the detuning of a laser field.
            This should be in units of frequency (MHz), and will be converted to detuning in angular frequency,
            :math:`\\delta(t)`, internally where needed, i.e. multiplied by :math:`2 \\pi`.
        wires (Union[int, List[int]]): integer or list containing wire values for the Rydberg atoms that
            the laser field acts on

    Returns:
        :class:`~.ParametrizedHamiltonian`: a :class:`~.ParametrizedHamiltonian` representing the action of the laser field on the Rydberg atoms.

    .. seealso::

        :func:`~.rydberg_interaction`, :class:`~.ParametrizedHamiltonian`, :class:`~.ParametrizedEvolution`
        and :func:`~.evolve`

    **Example**

    We create a Hamiltonian describing a laser acting on 4 wires (Rydberg atoms) with a fixed detuning and
    phase, and a parametrized, time-dependent amplitude. The Hamiltonian includes an interaction term for
    inter-atom interactions due to van der Waals forces, as well as the driving term for the laser driving
    the atoms.

    We provide all frequencies in the driving term in MHz (conversion to angular frequency, i.e. multiplication
    by :math:`2 \\pi`, is taken care of internally where needed). Phase (in radians) will not undergo unit conversion.

    For the driving field, we specify a detuning of
    :math:`\\delta = 1 \\times 2 \\pi \\text{MHz}`, and an
    amplitude :math:`\\Omega(t)` defined by a sinusoidal oscillation, squared to ensure a positve amplitude
    (a requirement for some hardware implementations). The maximum amplitude will dependent on the parameter ``p``
    passed to the amplitude function later, and should also be passed in units of MHz. We introduce a small phase
    shift as well, on the order of 1 rad.

    .. code-block:: python

        atom_coordinates = [[0, 0], [0, 4], [4, 0], [4, 4]]
        wires = [0, 1, 2, 3]
        H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires)

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t) ** 2
        phase = 0.25
        detuning = 1.
        H_d = qml.pulse.rydberg_drive(amplitude, phase, detuning, wires)

    >>> H_i
    HardwareHamiltonian: terms=6
    >>> H_d
    HardwareHamiltonian: terms=3

    The first two terms of the drive Hamiltonian ``H_d`` correspond to the first sum (the sine and cosine terms),
    describing drive between the ground and excited states. The third term corresponds to the shift term
    due to detuning from resonance. This drive term corresponds to a global drive that acts on all 4 wires of
    the device.

    The full Hamiltonian evolution and expectation value measurement can be executed in a ``QNode``:

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=wires)
        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H_i + H_d)(params, t=[0, 0.5])
            return qml.expval(qml.Z(0))

    Here we set a maximum amplitude of :math:`2.4 \\times 2 \\pi \\text{MHz}`, and calculate the result of running the pulse program:

    >>> params = [2.4]
    >>> circuit(params)
    Array(0.78301974, dtype=float64)
    >>> jax.grad(circuit)(params)
    [Array(-0.6250622, dtype=float64)]

    We can also create a Hamiltonian with local drives. The following circuit corresponds to the
    evolution where additional local drives acting on wires ``0`` and ``1`` respectively are added to the
    Hamiltonian:

    .. code-block:: python

        amplitude_local_0 = lambda p, t: p[0] * jnp.sin(2 * jnp.pi * t) ** 2 + p[1]
        phase_local_0 = jnp.pi / 4
        detuning_local_0 = lambda p, t: p * jnp.exp(-0.25 * t)
        H_local_0 = qml.pulse.rydberg_drive(amplitude_local_0, phase_local_0, detuning_local_0, [0])

        amplitude_local_1 = lambda p, t: jnp.cos(jnp.pi * t) ** 2 + p
        phase_local_1 = jnp.pi
        detuning_local_1 = lambda p, t: jnp.sin(jnp.pi * t) + p
        H_local_1 = qml.pulse.rydberg_drive(amplitude_local_1, phase_local_1, detuning_local_1, [1])

        H = H_i + H_d + H_local_0 + H_local_1

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 0.5])
            return qml.expval(qml.Z(0))

        p_global = 2.4
        p_local_amp_0 = [1.3, -2.0]
        p_local_det_0 = -1.5
        p_local_amp_1 = -0.9
        p_local_det_1 = 3.1
        params = [p_global, p_local_amp_0, p_local_det_0, p_local_amp_1, p_local_det_1]


    >>> circuit_local(params)
    Array(0.62640288, dtype=float64)
    >>> jax.grad(circuit_local)(params)
    [Array(1.07614151, dtype=float64),
     [Array(0.36370049, dtype=float64, weak_type=True),
      Array(0.91057498, dtype=float64, weak_type=True)],
     Array(1.3166343, dtype=float64),
     Array(-0.11102892, dtype=float64),
     Array(-0.02205843, dtype=float64)]
    """
    wires = Wires(wires)
    trivial_detuning = not callable(detuning) and qml.math.isclose(detuning, 0.0)
    if not callable(amplitude) and qml.math.isclose(amplitude, 0.0):
        if trivial_detuning:
            raise ValueError(f'Expected non-zero value for at least one of either amplitude or detuning, but received amplitude={amplitude} and detuning={detuning}. All terms are zero.')
        amplitude_term = HardwareHamiltonian([], [])
    else:
        amplitude_term = drive(amplitude, phase, wires)
    detuning_obs, detuning_coeffs = ([], [])
    if not trivial_detuning:
        detuning_obs.append(-0.5 * sum((qml.Identity(wire) for wire in wires)) * np.pi * 2 + 0.5 * sum((qml.Z(wire) for wire in wires)) * np.pi * 2)
        detuning_coeffs.append(detuning)
    detuning_term = HardwareHamiltonian(detuning_coeffs, detuning_obs)
    pulses = [HardwarePulse(amplitude, phase, detuning, wires)]
    drive_term = amplitude_term + detuning_term
    drive_term.pulses = pulses
    return drive_term