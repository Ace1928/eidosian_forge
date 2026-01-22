from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
Generate the processing function required to compute the vector-Jacobian products
            of a tape.

            This function can be used with multiple expectation values or a quantum state.
            When a quantum state is given,

            .. code-block:: python

                vjp_f = dev.vjp([qml.state()], grad_vec)
                vjp = vjp_f(tape)

            computes :math:`w = (w_1,\cdots,w_m)` where

            .. math::

                w_k = \langle v| \frac{\partial}{\partial \theta_k} | \psi_{\pmb{\theta}} \rangle.

            Here, :math:`m` is the total number of trainable parameters, :math:`\pmb{\theta}`
            is the vector of trainable parameters and :math:`\psi_{\pmb{\theta}}`
            is the output quantum state.

            Args:
                measurements (list): List of measurement processes for vector-Jacobian product.
                    Now it must be expectation values or a quantum state.
                grad_vec (tensor_like): Gradient-output vector. Must have shape matching the output
                    shape of the corresponding tape, i.e. number of measurements if
                    the return type is expectation or :math:`2^N` if the return type is statevector
                starting_state (tensor_like): post-forward pass state to start execution with.
                    It should be complex-valued. Takes precedence over ``use_device_state``.
                use_device_state (bool): use current device state to initialize.
                    A forward pass of the same circuit should be the last thing
                    the device has executed. If a ``starting_state`` is provided,
                    that takes precedence.

            Returns:
                The processing function required to compute the vector-Jacobian products of a tape.
            