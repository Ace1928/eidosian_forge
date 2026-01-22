def stirling_series(N):
    with mpmath.workdps(100):
        coeffs = [mpmath.bernoulli(2 * n) / (2 * n * (2 * n - 1)) for n in range(1, N + 1)]
    return coeffs