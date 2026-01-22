from typing import (
from numbers import Integral, Real
def iup_contour_optimize(deltas: _DeltaSegment, coords: _PointSegment, tolerance: Real=0.0) -> _DeltaOrNoneSegment:
    """For contour with coordinates `coords`, optimize a set of delta
    values `deltas` within error `tolerance`.

    Returns delta vector that has most number of None items instead of
    the input delta.
    """
    n = len(deltas)
    if all((abs(complex(*p)) <= tolerance for p in deltas)):
        return [None] * n
    if n == 1:
        return deltas
    d0 = deltas[0]
    if all((d0 == d for d in deltas)):
        return [d0] + [None] * (n - 1)
    forced = _iup_contour_bound_forced_set(deltas, coords, tolerance)
    if forced:
        k = n - 1 - max(forced)
        assert k >= 0
        deltas = _rot_list(deltas, k)
        coords = _rot_list(coords, k)
        forced = _rot_set(forced, k, n)
        chain, costs = _iup_contour_optimize_dp(deltas, coords, forced, tolerance)
        solution = set()
        i = n - 1
        while i is not None:
            solution.add(i)
            i = chain[i]
        solution.remove(-1)
        assert forced <= solution, (forced, solution)
        deltas = [deltas[i] if i in solution else None for i in range(n)]
        deltas = _rot_list(deltas, -k)
    else:
        chain, costs = _iup_contour_optimize_dp(deltas + deltas, coords + coords, forced, tolerance, n)
        best_sol, best_cost = (None, n + 1)
        for start in range(n - 1, len(costs) - 1):
            solution = set()
            i = start
            while i > start - n:
                solution.add(i % n)
                i = chain[i]
            if i == start - n:
                cost = costs[start] - costs[start - n]
                if cost <= best_cost:
                    best_sol, best_cost = (solution, cost)
        assert forced <= best_sol, (forced, best_sol)
        deltas = [deltas[i] if i in best_sol else None for i in range(n)]
    return deltas