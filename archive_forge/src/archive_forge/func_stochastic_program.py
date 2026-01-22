from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
def stochastic_program(self, if_optimize=True, objective_option='det', scale_nominal_param_value=False, scale_constant_value=1, optimize_opt=None, if_Cholesky=False, L_LB=1e-07, L_initial=None, jac_initial=None, fim_initial=None, formula='central', step=0.001, tee_opt=True):
    """
        Optimize DOE problem with design variables being the decisions.
        The DOE model is formed invasively and all scenarios are computed simultaneously.
        The function will first run a square problem with design variable being fixed at
        the given initial points (Objective function being 0), then a square problem with
        design variables being fixed at the given initial points (Objective function being Design optimality),
        and then unfix the design variable and do the optimization.

        Parameters
        ----------
        if_optimize:
            if true, continue to do optimization. else, just run square problem with given design variable values
        objective_option:
            choose from the ObjectiveLib enum,
            "det": maximizing the determinant with ObjectiveLib.det,
            "trace": or the trace of the FIM with ObjectiveLib.trace
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        optimize_opt:
            A dictionary, keys are design variables, values are True or False deciding if this design variable will be optimized as DOF or not
        if_Cholesky:
            if True, Cholesky decomposition is used for Objective function for D-optimality.
        L_LB:
            L is the Cholesky decomposition matrix for FIM, i.e. FIM = L*L.T.
            L_LB is the lower bound for every element in L.
            if FIM is positive definite, the diagonal element should be positive, so we can set a LB like 1E-10
        L_initial:
            initialize the L
        jac_initial:
            a matrix used to initialize jacobian matrix
        fim_initial:
            a matrix used to initialize FIM matrix
        formula:
            choose from "central", "forward", "backward",
            which refers to the Enum FiniteDifferenceStep.central, .forward, or .backward
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        tee_opt:
            if True, IPOPT console output is printed

        Returns
        -------
        analysis_square: result summary of the square problem solved at the initial point
        analysis_optimize: result summary of the optimization problem solved

        """
    self.design_values = self.design_vars.variable_names_value
    self.optimize = if_optimize
    self.objective_option = ObjectiveLib(objective_option)
    self.scale_nominal_param_value = scale_nominal_param_value
    self.scale_constant_value = scale_constant_value
    self.Cholesky_option = if_Cholesky
    self.L_LB = L_LB
    self.L_initial = L_initial
    self.jac_initial = jac_initial
    self.fim_initial = fim_initial
    self.formula = FiniteDifferenceStep(formula)
    self.step = step
    self.tee_opt = tee_opt
    self.fim_scale_constant_value = self.scale_constant_value ** 2
    sp_timer = TicTocTimer()
    sp_timer.tic(msg=None)
    m = self._create_doe_model(no_obj=True)
    m, analysis_square = self._compute_stochastic_program(m, optimize_opt)
    if self.optimize:
        analysis_optimize = self._optimize_stochastic_program(m)
        dT = sp_timer.toc(msg=None)
        self.logger.info('elapsed time: %0.1f' % dT)
        return (analysis_square, analysis_optimize)
    else:
        dT = sp_timer.toc(msg=None)
        self.logger.info('elapsed time: %0.1f' % dT)
        return analysis_square