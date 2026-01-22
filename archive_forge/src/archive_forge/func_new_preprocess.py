import contextlib
import pennylane as qml
from pennylane.operation import (
def new_preprocess(execution_config=qml.devices.DefaultExecutionConfig):
    program, config = original_preprocess(execution_config)
    for container in program:
        if container.transform == qml.devices.preprocess.decompose.transform:
            container.kwargs['decomposer'] = decomposer
            container.kwargs['max_expansion'] = decomp_depth
            for cond in ['stopping_condition', 'stopping_condition_shots']:
                original_stopping_condition = container.kwargs[cond]

                def stopping_condition(obj):
                    if obj.name in custom_decomps or type(obj) in custom_decomps:
                        return False
                    return original_stopping_condition(obj)
                container.kwargs[cond] = stopping_condition
            break
    return (program, config)