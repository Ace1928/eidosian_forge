from ase.calculators.calculator import Parameters
def format_value(value):
    """
    Format python values to fdf-format.

    Parameters:
        - value : The value to format.
    """
    if isinstance(value, tuple):
        sub_values = [format_value(v) for v in value]
        value = '\t'.join(sub_values)
    elif isinstance(value, list):
        sub_values = [format_value(v) for v in value]
        value = '\n'.join(sub_values)
    else:
        value = str(value)
    return value