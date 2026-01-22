from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import devices, ops, protocols
def decompose_cphase_into_two_fsim(cphase_gate: 'cirq.CZPowGate', *, fsim_gate: 'cirq.FSimGate', qubits: Optional[Sequence['cirq.Qid']]=None, atol: float=1e-08) -> 'cirq.OP_TREE':
    """Decomposes CZPowGate into two FSimGates.

    This function implements the decomposition described in section VII F I
    of https://arxiv.org/abs/1910.11333.

    The decomposition results in exactly two FSimGates and a few single-qubit
    rotations. It is feasible if and only if one of the following conditions
    is met:

        |sin(θ)| <= |sin(δ/4)| <= |sin(φ/2)|
        |sin(φ/2)| <= |sin(δ/4)| <= |sin(θ)|

    where:

         θ = fsim_gate.theta,
         φ = fsim_gate.phi,
         δ = -π * cphase_gate.exponent.

    Note that the gate parameterizations are non-injective. For the
    decomposition to be feasible it is sufficient that one of the
    parameter values which correspond to the provided gate satisfies
    the constraints. This function will find and use the appropriate
    value whenever it exists.

    The constraints above imply that certain FSimGates are not suitable
    for use in this decomposition regardless of the target CZPowGate. We
    reject such gates based on how close |sin(θ)| is to |sin(φ/2)|, see
    atol argument below.

    This implementation accounts for the global phase.

    Args:
        cphase_gate: The CZPowGate to synthesize.
        fsim_gate: The only two qubit gate that is permitted to appear in the
            output.
        qubits: The qubits to apply the resulting operations to. If not set,
            defaults `cirq.LineQubit.range(2)`.
        atol: Tolerance used to determine whether fsim_gate is valid. The gate
            is invalid if the squares of the sines of the theta angle and half
            the phi angle are too close.

    Returns:
        Operations equivalent to cphase_gate and consisting solely of two copies
        of fsim_gate and a few single-qubit rotations.

    Raises:
        ValueError: Under any of the following circumstances:
            * cphase_gate or fsim_gate is parametrized,
            * cphase_gate and fsim_gate do not satisfy the conditions above,
            * fsim_gate has invalid angles (see atol argument above),
            * incorrect number of qubits are provided.
    """
    if protocols.is_parameterized(cphase_gate):
        raise ValueError('Cannot decompose a parametrized gate.')
    if protocols.is_parameterized(fsim_gate):
        raise ValueError('Cannot decompose into a parametrized gate.')
    if qubits is None:
        qubits = devices.LineQubit.range(2)
    if len(qubits) != 2:
        raise ValueError(f'Expected a pair of qubits, but got {qubits!r}.')
    q0, q1 = qubits
    theta = fsim_gate.theta
    phi = fsim_gate.phi
    assert isinstance(theta, float) and isinstance(phi, float)
    sin_half_phi = np.sin(phi / 2)
    cos_half_phi = np.cos(phi / 2)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = (sin_theta - sin_half_phi) * (sin_theta + sin_half_phi)
    if abs(denominator) < atol:
        raise ValueError(f'{fsim_gate} cannot be used to decompose CZPowGate because sin(theta)**2 is too close to sin(phi/2)**2 (difference is {denominator}).')
    canonical_delta = -np.pi * (cphase_gate.exponent % 2)
    for delta in (canonical_delta, canonical_delta + 2 * np.pi):
        sin_quarter_delta = np.sin(delta / 4)
        numerator = (sin_quarter_delta - sin_half_phi) * (sin_quarter_delta + sin_half_phi)
        sin_alpha_squared = numerator / denominator
        if 0 <= sin_alpha_squared <= 1:
            break
    else:
        intervals = compute_cphase_exponents_for_fsim_decomposition(fsim_gate)
        raise ValueError(f'{cphase_gate} cannot be decomposed into two {fsim_gate}. Valid intervals for canonical exponent of CZPowGate: {intervals}.')
    assert 0 <= sin_alpha_squared <= 1
    alpha = np.arcsin(np.sqrt(sin_alpha_squared))
    tan_alpha = np.tan(alpha)
    xi = np.arctan2(tan_alpha * cos_theta, cos_half_phi)
    eta = np.arctan2(tan_alpha * sin_theta, sin_half_phi)
    if delta < 0:
        eta += np.pi
    return (ops.rx(xi).on(q0), ops.rx(eta).on(q1), fsim_gate.on(q0, q1), ops.rz(phi / 2).on(q0), ops.rz(phi / 2).on(q1), ops.global_phase_operation(np.exp(1j * phi / 4)), ops.rx(-2 * alpha).on(q0), ops.Z(q0), fsim_gate.on(q0, q1), ops.rz(phi / 2).on(q0), ops.rz(phi / 2).on(q1), ops.global_phase_operation(np.exp(1j * phi / 4)), ops.Z(q0), ops.rx(-eta).on(q1), ops.rx(xi).on(q0), ops.rz(-delta / 2).on(q0), ops.rz(-delta / 2).on(q1), ops.global_phase_operation(np.exp(-1j * delta / 4)))