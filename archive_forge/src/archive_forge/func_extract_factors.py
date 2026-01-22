from enum import Enum
@staticmethod
def extract_factors(string_factors):
    string_factors = string_factors.strip()
    if not string_factors:
        return {}
    list_factors = []
    eval_factor_units = string_factors.split(';')
    for eval_factor_unit in eval_factor_units:
        try:
            factor = int(eval_factor_unit)
            list_factors.append(factor)
        except ValueError:
            string_bounds = eval_factor_unit.split('-')
            if len(string_bounds) != 2:
                raise AttributeError('Range need to contain exactly two numbers!')
            begin_range = int(string_bounds[0])
            end_range = int(string_bounds[1])
            list_factors += list(range(begin_range, end_range + 1))
    return set(list_factors)