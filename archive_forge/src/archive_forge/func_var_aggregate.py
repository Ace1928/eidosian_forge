from __future__ import annotations
def var_aggregate(x, ddof):
    squares, totals, counts = list(zip(*x))
    x2, x, n = (float(sum(squares)), float(sum(totals)), sum(counts))
    result = x2 / n - (x / n) ** 2
    return result * n / (n - ddof)