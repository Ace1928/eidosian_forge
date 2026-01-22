def taylor_series_at_1(N):
    coeffs = []
    with mpmath.workdps(100):
        coeffs.append(-mpmath.euler)
        for n in range(2, N + 1):
            coeffs.append((-1) ** n * mpmath.zeta(n) / n)
    return coeffs