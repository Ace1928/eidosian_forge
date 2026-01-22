import sys
def is_stdlib_module(module: str) -> bool:
    base_module = module.partition('.')[0]
    return base_module in _get_stdlib_modules()