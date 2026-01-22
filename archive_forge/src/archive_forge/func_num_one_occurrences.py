from collections import Counter
from math import log, sqrt
def num_one_occurrences(observed_obj_vals, tolerance):
    """
    Determines the number of optima that have only been observed once.
    Needed to estimate missing mass of optima.
    """
    obj_value_distribution = Counter(observed_obj_vals)
    sorted_histogram = list(sorted(obj_value_distribution.items()))
    if tolerance == 0:
        return sum((1 for _, count in sorted_histogram if count == 1))
    else:
        num_obj_vals_only_observed_once = 0
        for i, tup in enumerate(sorted_histogram):
            obj_val, count = tup
            if count == 1:
                if i > 0 and obj_val - sorted_histogram[i - 1][0] <= tolerance:
                    continue
                if i < len(sorted_histogram) - 1 and sorted_histogram[i + 1][0] - obj_val <= tolerance:
                    continue
                num_obj_vals_only_observed_once += 1
        return num_obj_vals_only_observed_once