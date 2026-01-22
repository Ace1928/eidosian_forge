import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ZeroPrior
Given a set of observations, X, Y; optimize the scale
        of the Gaussian Process maximizing the marginal log-likelihood.
        This method calls TRAIN there is no need to call the TRAIN method
        again. The method also sets the parameters of the Kernel to their
        optimal value at the end of execution

        Parameters:

        X:   observations(i.e. positions). numpy array with shape: nsamples x D
        Y:   targets (i.e. energy and forces).
             numpy array with shape (nsamples, D+1)
        tol: tolerance on the maximum component of the gradient of the
             log-likelihood.
             (See scipy's L-BFGS-B documentation:
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        eps: include bounds to the hyperparameters as a +- a percentage
             if eps is None there are no bounds in the optimization

        Returns:

        result (dict) :
              result = {'hyperparameters': (numpy.array) New hyperparameters,
                        'converged': (bool) True if it converged,
                                            False otherwise
                       }
        