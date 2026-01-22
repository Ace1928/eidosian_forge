from typing import Any, Tuple
def get_module_from_name(module, tensor_name: str) -> Tuple[Any, str]:
    if '.' in tensor_name:
        splits = tensor_name.split('.')
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f'{module} has no attribute {split}.')
            module = new_module
        tensor_name = splits[-1]
    return (module, tensor_name)