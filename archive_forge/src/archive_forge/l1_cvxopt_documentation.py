import numpy as np
import statsmodels.base.l1_solvers_common as l1_solvers_common

    Wraps the hessian up in the form for cvxopt.

    cvxopt wants the hessian of the objective function and the constraints.
        Since our constraints are linear, this part is all zeros.
    