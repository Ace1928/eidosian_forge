from __future__ import absolute_import, division, print_function
import re
def parse_number_with_metric_suffix(module, number, factor=1024):
    """Given a human-readable string (e.g. 2G, 30M, 400),
    return the resolved integer.
    Will call `module.fail_json()` for invalid inputs.
    """
    try:
        stripped_num = number.strip()
        if stripped_num[-1].isdigit():
            return int(stripped_num)
        result = float(stripped_num[:-1])
        suffix = stripped_num[-1].upper()
        factor_count = METRIC_SUFFIXES.index(suffix) + 1
        for _i in range(0, factor_count):
            result = result * float(factor)
        return int(result)
    except Exception:
        module.fail_json(msg="'{0}' is not a valid number, use '400', '1K', '2M', ...".format(number))
    return 0