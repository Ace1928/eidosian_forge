from copy import copy
from typing import Any, Dict, Optional, Sequence, Callable
from pennylane import apply, adjoint
from pennylane.math import mean, shape, round
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms import transform
import pennylane as qml
@transform
def mitigate_with_zne(tape: QuantumTape, scale_factors: Sequence[float], folding: callable, extrapolate: callable, folding_kwargs: Optional[Dict[str, Any]]=None, extrapolate_kwargs: Optional[Dict[str, Any]]=None, reps_per_factor=1) -> (Sequence[QuantumTape], Callable):
    """Mitigate an input circuit using zero-noise extrapolation.

    Error mitigation is a precursor to error correction and is compatible with near-term quantum
    devices. It aims to lower the impact of noise when evaluating a circuit on a quantum device by
    evaluating multiple variations of the circuit and post-processing the results into a
    noise-reduced estimate. This transform implements the zero-noise extrapolation (ZNE) method
    originally introduced by
    `Temme et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`__ and
    `Li et al. <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.021050>`__.

    Details on the functions passed to the ``folding`` and ``extrapolate`` arguments of this
    transform can be found in the usage details. This transform is compatible with functionality
    from the `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package (version 0.11.0 and above),
    see the example and usage details for further information.

    Args:
        tape (QNode or QuantumTape): the quantum circuit to be error-mitigated
        scale_factors (Sequence[float]): the range of noise scale factors used
        folding (callable): a function that returns a folded circuit for a specified scale factor
        extrapolate (callable): a function that returns an extrapolated result when provided a
            range of scale factors and corresponding results
        folding_kwargs (dict): optional keyword arguments passed to the ``folding`` function
        extrapolate_kwargs (dict): optional keyword arguments passed to the ``extrapolate`` function
        reps_per_factor (int): Number of circuits generated for each scale factor. Useful when the
            folding function is stochastic.

    Returns:
        qnode (QNode) or tuple[List[.QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the mitigated results in the form of a tensor of a tensor, a tuple, or a nested tuple depending
        upon the nesting structure of measurements in the original circuit.

    **Example:**

    We first create a noisy device using ``default.mixed`` by adding :class:`~.AmplitudeDamping` to
    each gate of circuits executed on the device using the :func:`~.transforms.insert` transform:

    .. code-block:: python3

        import pennylane as qml

        noise_strength = 0.05

        dev = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev)

    We can now set up a mitigated QNode by passing a ``folding`` and ``extrapolate`` function. PennyLane provides native
    functions :func:`~.pennylane.transforms.fold_global` and :func:`~.pennylane.transforms.poly_extrapolate` or :func:`~.pennylane.transforms.richardson_extrapolate` that
    allow for differentiating through them. Custom functions, as well as functionalities from the `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package
    are supported as well (see usage details below).

    .. code-block:: python3

        from functools import partial
        from pennylane import numpy as np
        from pennylane import qnode

        from pennylane.transforms import fold_global, poly_extrapolate

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @partial(qml.transforms.mitigate_with_zne, [1., 2., 3.], fold_global, poly_extrapolate, extrapolate_kwargs={'order': 2})
        @qnode(dev)
        def circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.Z(0))

    Executions of ``circuit`` will now be mitigated:

    >>> circuit(w1, w2)
    0.19113067083636542

    The unmitigated circuit result is ``0.33652776`` while the ideal circuit result is
    ``0.23688169`` and we can hence see that mitigation has helped reduce our estimation error.

    This mitigated qnode can be differentiated like any other qnode.

    >>> qml.grad(circuit)(w1, w2)
    (array([-0.89319941,  0.37949841]),
     array([[[-7.04121596e-01,  3.00073104e-01]],
            [[-6.41155176e-01,  8.32667268e-17]]]))

    .. details::
        :title: Usage Details

        **Theoretical details**

        A summary of ZNE can be found in `LaRose et al. <https://arxiv.org/abs/2009.04417>`__. The
        method works by assuming that the amount of noise present when a circuit is run on a
        noisy device is enumerated by a parameter :math:`\\gamma`. Suppose we have an input circuit
        that experiences an amount of noise equal to :math:`\\gamma = \\gamma_{0}` when executed.
        Ideally, we would like to evaluate the result of the circuit in the :math:`\\gamma = 0`
        noise-free setting.

        To do this, we create a family of equivalent circuits whose ideal noise-free value is the
        same as our input circuit. However, when run on a noisy device, each circuit experiences
        a noise equal to :math:`\\gamma = s \\gamma_{0}` for some scale factor :math:`s`. By
        evaluating the noisy outputs of each circuit, we can extrapolate to :math:`s=0` to estimate
        the result of running a noise-free circuit.

        A key element of ZNE is the ability to run equivalent circuits for a range of scale factors
        :math:`s`. When the noise present in a circuit scales with the number of gates, :math:`s`
        can be varied using `unitary folding <https://ieeexplore.ieee.org/document/9259940>`__.
        Unitary folding works by noticing that a unitary :math:`U` is equivalent to
        :math:`U U^{\\dagger} U`. This type of transform can be applied to individual gates in the
        circuit or to the whole circuit. When no folding occurs, the scale factor is
        :math:`s=1` and we are running our input circuit. On the other hand, when each gate has been
        folded once, we have tripled the amount of noise in the circuit so that :math:`s=3`. For
        :math:`s \\geq 3`, each gate in the circuit will be folded more than once. A typical choice
        of scale parameters is :math:`(1, 2, 3)`.

        **Unitary folding**

        This transform applies ZNE to an input circuit using the unitary folding approach. It
        requires a callable to be passed as the ``folding`` argument with signature

        .. code-block:: python

            fn(circuit, scale_factor, **folding_kwargs)

        where

        - ``circuit`` is a quantum tape,

        - ``scale_factor`` is a float, and

        - ``folding_kwargs`` are optional keyword arguments.

        The output of the function should be the folded circuit as a quantum tape.
        Folding functionality is available from the
        `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package (version 0.11.0 and above)
        in the
        `zne.scaling.folding <https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.scaling.folding>`__
        module.

        .. warning::

            Calculating the gradient of mitigated circuits is not supported when using the Mitiq
            package as a backend for folding or extrapolation.

        **Extrapolation**

        This transform also requires a callable to be passed to the ``extrapolate`` argument that
        returns the extrapolated value(s). Its function should be

        .. code-block:: python

            fn(scale_factors, results, **extrapolate_kwargs)

        where

        - ``scale_factors`` are the ZNE scale factors,

        - ``results`` are the execution results of the circuit at the specified scale
          factors of shape ``(len(scale_factors), len(qnode_returns))``, and

        - ``extrapolate_kwargs`` are optional keyword arguments.

        The output of the extrapolate ``fn`` should be a flat array of
        length ``len(qnode_returns)``.

        Extrapolation functionality is available using ``extrapolate``
        methods of the factories in the
        `mitiq.zne.inference <https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.inference>`__
        module.
    """
    folding_kwargs = folding_kwargs or {}
    extrapolate_kwargs = extrapolate_kwargs or {}
    tape = tape.expand(stop_at=lambda op: not isinstance(op, QuantumScript))
    script_removed = QuantumScript(tape.operations[tape.num_preps:])
    tapes = [[folding(script_removed, s, **folding_kwargs) for _ in range(reps_per_factor)] for s in scale_factors]
    tapes = [tape_ for tapes_ in tapes for tape_ in tapes_]
    if isinstance(tapes[0], tuple) and isinstance(tapes[0][0], list) and callable(tapes[0][1]):
        tapes = [t[0] for t, _ in tapes]
    prep_ops = tape.operations[:tape.num_preps]
    out_tapes = [QuantumScript(prep_ops + tape_.operations, tape.measurements) for tape_ in tapes]

    def processing_fn(results):
        """Maps from input tape executions to an error-mitigated estimate"""
        results = list(results)
        for i, tape in enumerate(out_tapes):
            if len(tape.observables) > 1:
                results[i] = qml.math.stack(results[i])
        results_flattened = []
        for i in range(0, len(results), reps_per_factor):
            results_flattened.append(mean(qml.math.stack(results[i:i + reps_per_factor]), axis=0))
        extrapolated = extrapolate(scale_factors, results_flattened, **extrapolate_kwargs)
        extrapolated = extrapolated[0] if shape(extrapolated) == (1,) else extrapolated
        return extrapolated if shape(extrapolated) == () else tuple(qml.math.unstack(extrapolated))
    return (out_tapes, processing_fn)