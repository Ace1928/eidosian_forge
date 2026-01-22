from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def findSolutionValues(self, lp):
    CplexLpStatus = {lp.solverModel.solution.status.MIP_optimal: constants.LpStatusOptimal, lp.solverModel.solution.status.optimal: constants.LpStatusOptimal, lp.solverModel.solution.status.optimal_tolerance: constants.LpStatusOptimal, lp.solverModel.solution.status.infeasible: constants.LpStatusInfeasible, lp.solverModel.solution.status.infeasible_or_unbounded: constants.LpStatusInfeasible, lp.solverModel.solution.status.MIP_infeasible: constants.LpStatusInfeasible, lp.solverModel.solution.status.MIP_infeasible_or_unbounded: constants.LpStatusInfeasible, lp.solverModel.solution.status.unbounded: constants.LpStatusUnbounded, lp.solverModel.solution.status.MIP_unbounded: constants.LpStatusUnbounded, lp.solverModel.solution.status.abort_dual_obj_limit: constants.LpStatusNotSolved, lp.solverModel.solution.status.abort_iteration_limit: constants.LpStatusNotSolved, lp.solverModel.solution.status.abort_obj_limit: constants.LpStatusNotSolved, lp.solverModel.solution.status.abort_relaxed: constants.LpStatusNotSolved, lp.solverModel.solution.status.abort_time_limit: constants.LpStatusNotSolved, lp.solverModel.solution.status.abort_user: constants.LpStatusNotSolved, lp.solverModel.solution.status.MIP_abort_feasible: constants.LpStatusOptimal, lp.solverModel.solution.status.MIP_time_limit_feasible: constants.LpStatusOptimal, lp.solverModel.solution.status.MIP_time_limit_infeasible: constants.LpStatusInfeasible}
    lp.cplex_status = lp.solverModel.solution.get_status()
    status = CplexLpStatus.get(lp.cplex_status, constants.LpStatusUndefined)
    CplexSolStatus = {lp.solverModel.solution.status.MIP_time_limit_feasible: constants.LpSolutionIntegerFeasible, lp.solverModel.solution.status.MIP_abort_feasible: constants.LpSolutionIntegerFeasible, lp.solverModel.solution.status.MIP_feasible: constants.LpSolutionIntegerFeasible}
    sol_status = CplexSolStatus.get(lp.cplex_status)
    lp.assignStatus(status, sol_status)
    var_names = [var.name for var in lp._variables]
    con_names = [con for con in lp.constraints]
    try:
        objectiveValue = lp.solverModel.solution.get_objective_value()
        variablevalues = dict(zip(var_names, lp.solverModel.solution.get_values(var_names)))
        lp.assignVarsVals(variablevalues)
        constraintslackvalues = dict(zip(con_names, lp.solverModel.solution.get_linear_slacks(con_names)))
        lp.assignConsSlack(constraintslackvalues)
        if lp.solverModel.get_problem_type() == cplex.Cplex.problem_type.LP:
            variabledjvalues = dict(zip(var_names, lp.solverModel.solution.get_reduced_costs(var_names)))
            lp.assignVarsDj(variabledjvalues)
            constraintpivalues = dict(zip(con_names, lp.solverModel.solution.get_dual_values(con_names)))
            lp.assignConsPi(constraintpivalues)
    except cplex.exceptions.CplexSolverError:
        pass
    if self.msg:
        print('Cplex status=', lp.cplex_status)
    lp.resolveOK = True
    for var in lp._variables:
        var.isModified = False
    return status