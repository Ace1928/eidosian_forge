import math
def cost_node(x, allow_missed=True):
    f = x.fn_name
    if f in COSTS:
        return COSTS[f](x)
    elif allow_missed:
        return 0
    else:
        raise ValueError(f'Cost for {f} not implemented.')