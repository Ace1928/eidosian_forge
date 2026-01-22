import warnings
from string import ascii_letters as ABC
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
def default_qnode_wrapper(self, qnode, targs, tkwargs):
    hybrid = tkwargs.pop('hybrid', self.hybrid)
    argnums = tkwargs.get('argnums', None)
    old_interface = qnode.interface
    _wrapper = super().default_qnode_wrapper(qnode, targs, tkwargs)
    cjac_fn = qml.gradients.classical_jacobian(qnode, argnum=argnums, expand_fn=self.expand_fn)

    def hessian_wrapper(*args, **kwargs):
        if argnums is not None:
            argnums_ = [argnums] if isinstance(argnums, int) else argnums
            params = qml.math.jax_argnums_to_tape_trainable(qnode, argnums_, self.expand_fn, args, kwargs)
            argnums_ = qml.math.get_trainable_indices(params)
            kwargs['argnums'] = argnums_
        if not qml.math.get_trainable_indices(args) and (not argnums):
            warnings.warn('Attempted to compute the Hessian of a QNode with no trainable parameters. If this is unintended, please add trainable parameters in accordance with the chosen auto differentiation framework.')
            return ()
        qhess = _wrapper(*args, **kwargs)
        if old_interface == 'auto':
            qnode.interface = 'auto'
        if not hybrid:
            return qhess
        if len(qnode.tape.measurements) == 1:
            qhess = (qhess,)
        kwargs.pop('shots', False)
        if argnums is None and qml.math.get_interface(*args) == 'jax':
            cjac = qml.gradients.classical_jacobian(qnode, argnum=qml.math.get_trainable_indices(args), expand_fn=self.expand_fn)(*args, **kwargs)
        else:
            cjac = cjac_fn(*args, **kwargs)
        has_single_arg = False
        if not isinstance(cjac, tuple):
            has_single_arg = True
            cjac = (cjac,)
        hessians = []
        for jac in cjac:
            if jac is not None:
                hess = _process_jacs(jac, qhess)
                hessians.append(hess)
        return hessians[0] if has_single_arg else tuple(hessians)
    return hessian_wrapper