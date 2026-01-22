import inspect
import pennylane as qml
def specs_qnode(*args, **kwargs):
    """Returns information on the structure and makeup of provided QNode.

        Dictionary keys:
            * ``"num_operations"`` number of operations in the qnode
            * ``"num_observables"`` number of observables in the qnode
            * ``"num_diagonalizing_gates"`` number of diagonalizing gates required for execution of the qnode
            * ``"resources"``: a :class:`~.resource.Resources` object containing resource quantities used by the qnode
            * ``"num_used_wires"``: number of wires used by the circuit
            * ``"num_device_wires"``: number of wires in device
            * ``"depth"``: longest path in directed acyclic graph representation
            * ``"device_name"``: name of QNode device
            * ``"expansion_strategy"``: string specifying method for decomposing operations in the circuit
            * ``"gradient_options"``: additional configurations for gradient computations
            * ``"interface"``: autodiff framework to dispatch to for the qnode execution
            * ``"diff_method"``: a string specifying the differntiation method
            * ``"gradient_fn"``: executable to compute the gradient of the qnode

        Potential Additional Information:
            * ``"num_trainable_params"``: number of individual scalars that are trainable
            * ``"num_gradient_executions"``: number of times circuit will execute when
                    calculating the derivative

        Returns:
            dict[str, Union[defaultdict,int]]: dictionaries that contain QNode specifications
        """
    initial_max_expansion = qnode.max_expansion
    initial_expansion_strategy = getattr(qnode, 'expansion_strategy', None)
    try:
        qnode.max_expansion = initial_max_expansion if max_expansion is None else max_expansion
        qnode.expansion_strategy = expansion_strategy or initial_expansion_strategy
        qnode.construct(args, kwargs)
    finally:
        qnode.max_expansion = initial_max_expansion
        qnode.expansion_strategy = initial_expansion_strategy
    info = qnode.qtape.specs.copy()
    info['num_device_wires'] = len(qnode.tape.wires) if isinstance(qnode.device, qml.devices.Device) else len(qnode.device.wires)
    info['device_name'] = getattr(qnode.device, 'short_name', qnode.device.name)
    info['expansion_strategy'] = qnode.expansion_strategy
    info['gradient_options'] = qnode.gradient_kwargs
    info['interface'] = qnode.interface
    info['diff_method'] = _get_absolute_import_path(qnode.diff_method) if callable(qnode.diff_method) else qnode.diff_method
    if isinstance(qnode.gradient_fn, qml.transforms.core.TransformDispatcher):
        info['gradient_fn'] = _get_absolute_import_path(qnode.gradient_fn)
        try:
            info['num_gradient_executions'] = len(qnode.gradient_fn(qnode.qtape)[0])
        except Exception as e:
            info['num_gradient_executions'] = f'NotSupported: {str(e)}'
    else:
        info['gradient_fn'] = qnode.gradient_fn
    return info