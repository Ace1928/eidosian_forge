from itertools import product
import os
import numpy as np
def pss_block(seed, k, case, i1, block_id, m=2000000, t=1000, save=True, path='./'):
    file_name = f'pss-k-{k}-case-{case}-i1-{i1}-block-{block_id}.npz'
    file_name = os.path.join(path, file_name)
    if save and os.path.exists(file_name):
        return
    rs = np.random.default_rng(seed)
    const = np.ones(t - 1)
    tau = np.arange(1, t).astype(float)
    f = np.empty(m)
    for j in range(m):
        u = rs.standard_normal((k + 1, t))
        y = np.cumsum(u[0])
        if i1:
            x = np.cumsum(u[1:], axis=1).T
        else:
            x = u[1:].T
        lhs = np.diff(y)
        rhv = [y[:-1], x[:-1]]
        if case == 2:
            rhv.append(const)
        elif case == 4:
            rhv.append(tau)
        if case >= 3:
            rhv.append(const)
        if case == 5:
            rhv.append(tau)
        rest = k + 1
        if case in (2, 4):
            rest += 1
        rhs = np.column_stack(rhv)
        b = np.linalg.lstsq(rhs, lhs, rcond=None)[0]
        u = lhs - rhs @ b
        s2 = u.T @ u / (u.shape[0] - rhs.shape[1])
        xpx = rhs.T @ rhs
        vcv = np.linalg.inv(xpx) * s2
        r = np.eye(rest, rhs.shape[1])
        rvcvr = r @ vcv @ r.T
        rb = r @ b
        f[j] = rb.T @ np.linalg.inv(rvcvr) @ rb / rest
    percentiles = [0.05]
    percentiles += [i / 10 for i in range(1, 10)]
    percentiles += [1 + i / 2 for i in range(18)]
    percentiles += [i for i in range(10, 51)]
    percentiles += [100 - v for v in percentiles]
    percentiles = sorted(set(percentiles))
    percentiles = np.asarray(percentiles)
    q = np.percentile(f, percentiles)
    if save:
        np.savez(file_name, q=q, percentiles=percentiles)
    return q