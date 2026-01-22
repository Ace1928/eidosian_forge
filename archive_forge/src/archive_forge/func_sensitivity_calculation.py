from pyomo.environ import (
from pyomo.common.sorting import sorted_robust
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface, InTempDir
import logging
import os
import io
import shutil
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy, scipy_available
def sensitivity_calculation(method, instance, paramList, perturbList, cloneModel=True, tee=False, keepfiles=False, solver_options=None):
    """This function accepts a Pyomo ConcreteModel, a list of parameters, and
    their corresponding perturbation list. The model is then augmented with
    dummy constraints required to call sipopt or k_aug to get an approximation
    of the perturbed solution.

    Parameters
    ----------
    method: string
        'sipopt' or 'k_aug'
    instance: Block
        pyomo block or model object
    paramSubList: list
        list of mutable parameters or fixed variables
    perturbList: list
        list of perturbed parameter values
    cloneModel: bool, optional
        indicator to clone the model. If set to False, the original
        model will be altered
    tee: bool, optional
        indicator to stream solver log
    keepfiles: bool, optional
        preserve solver interface files
    solver_options: dict, optional
        Provides options to the solver (also the name of an attribute)

    Returns
    -------
    The model that was manipulated by the sensitivity interface

    """
    sens = SensitivityInterface(instance, clone_model=cloneModel)
    sens.setup_sensitivity(paramList)
    m = sens.model_instance
    if method not in {'k_aug', 'sipopt'}:
        raise ValueError("Only methods 'k_aug' and 'sipopt' are supported'")
    if method == 'k_aug':
        k_aug = SolverFactory('k_aug', solver_io='nl')
        dot_sens = SolverFactory('dot_sens', solver_io='nl')
        ipopt = SolverFactory('ipopt', solver_io='nl')
        k_aug_interface = K_augInterface(k_aug=k_aug, dot_sens=dot_sens)
        ipopt.solve(m, tee=tee)
        m.ipopt_zL_in.update(m.ipopt_zL_out)
        m.ipopt_zU_in.update(m.ipopt_zU_out)
        k_aug.options['dsdp_mode'] = ''
        k_aug_interface.k_aug(m, tee=tee)
    sens.perturb_parameters(perturbList)
    if method == 'sipopt':
        ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')
        ipopt_sens.options['run_sens'] = 'yes'
        if solver_options is not None:
            ipopt_sens.options['linear_solver'] = solver_options
        results = ipopt_sens.solve(m, keepfiles=keepfiles, tee=tee)
    elif method == 'k_aug':
        dot_sens.options['dsdp_mode'] = ''
        k_aug_interface.dot_sens(m, tee=tee)
    return m