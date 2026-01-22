from enum import Enum
@staticmethod
def group_factors_by_range(factors_set):
    factors = sorted(list(factors_set))
    is_start = []
    for i in range(0, len(factors)):
        is_start.append(i == 0 or factors[i] != factors[i - 1] + 1)
    grouped_factors = []
    i = 0
    while i < len(factors):
        if is_start[i]:
            grouped_factors.append([])
        grouped_factors[-1].append(factors[i])
        i += 1
    return grouped_factors