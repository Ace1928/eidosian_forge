import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def eval_evec(symmetric, d, typ, k, which, v0=None, sigma=None, mattype=np.asarray, OPpart=None, mode='normal'):
    general = 'bmat' in d
    if symmetric:
        eigs_func = eigsh
    else:
        eigs_func = eigs
    if general:
        err = 'error for {}:general, typ={}, which={}, sigma={}, mattype={}, OPpart={}, mode={}'.format(eigs_func.__name__, typ, which, sigma, mattype.__name__, OPpart, mode)
    else:
        err = 'error for {}:standard, typ={}, which={}, sigma={}, mattype={}, OPpart={}, mode={}'.format(eigs_func.__name__, typ, which, sigma, mattype.__name__, OPpart, mode)
    a = d['mat'].astype(typ)
    ac = mattype(a)
    if general:
        b = d['bmat'].astype(typ)
        bc = mattype(b)
    exact_eval = d['eval'].astype(typ.upper())
    ind = argsort_which(exact_eval, typ, k, which, sigma, OPpart, mode)
    exact_eval = exact_eval[ind]
    kwargs = dict(which=which, v0=v0, sigma=sigma)
    if eigs_func is eigsh:
        kwargs['mode'] = mode
    else:
        kwargs['OPpart'] = OPpart
    kwargs['tol'], rtol, atol = _get_test_tolerance(typ, mattype, d, which)
    ntries = 0
    while ntries < 5:
        if general:
            try:
                eigenvalues, evec = eigs_func(ac, k, bc, **kwargs)
            except ArpackNoConvergence:
                kwargs['maxiter'] = 20 * a.shape[0]
                eigenvalues, evec = eigs_func(ac, k, bc, **kwargs)
        else:
            try:
                eigenvalues, evec = eigs_func(ac, k, **kwargs)
            except ArpackNoConvergence:
                kwargs['maxiter'] = 20 * a.shape[0]
                eigenvalues, evec = eigs_func(ac, k, **kwargs)
        ind = argsort_which(eigenvalues, typ, k, which, sigma, OPpart, mode)
        eigenvalues = eigenvalues[ind]
        evec = evec[:, ind]
        try:
            assert_allclose_cc(eigenvalues, exact_eval, rtol=rtol, atol=atol, err_msg=err)
            check_evecs = True
        except AssertionError:
            check_evecs = False
            ntries += 1
        if check_evecs:
            LHS = np.dot(a, evec)
            if general:
                RHS = eigenvalues * np.dot(b, evec)
            else:
                RHS = eigenvalues * evec
            assert_allclose(LHS, RHS, rtol=rtol, atol=atol, err_msg=err)
            break
    assert_allclose_cc(eigenvalues, exact_eval, rtol=rtol, atol=atol, err_msg=err)