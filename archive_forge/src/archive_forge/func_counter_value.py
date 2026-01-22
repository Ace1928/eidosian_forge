import torch._C._lazy
def counter_value(name: str):
    """Return the value of the counter with the speficied name"""
    return torch._C._lazy._counter_value(name)