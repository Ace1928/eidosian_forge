from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def buildSolverModel(self, lp):
    """
            Takes the pulp lp model and translates it into a cplex model
            """
    model_variables = lp.variables()
    self.n2v = {var.name: var for var in model_variables}
    if len(self.n2v) != len(model_variables):
        raise PulpSolverError('Variables must have unique names for cplex solver')
    log.debug('create the cplex model')
    self.solverModel = lp.solverModel = cplex.Cplex()
    log.debug('set the name of the problem')
    if not self.mip:
        self.solverModel.set_problem_name(lp.name)
    log.debug('set the sense of the problem')
    if lp.sense == constants.LpMaximize:
        lp.solverModel.objective.set_sense(lp.solverModel.objective.sense.maximize)
    obj = [float(lp.objective.get(var, 0.0)) for var in model_variables]

    def cplex_var_lb(var):
        if var.lowBound is not None:
            return float(var.lowBound)
        else:
            return -cplex.infinity
    lb = [cplex_var_lb(var) for var in model_variables]

    def cplex_var_ub(var):
        if var.upBound is not None:
            return float(var.upBound)
        else:
            return cplex.infinity
    ub = [cplex_var_ub(var) for var in model_variables]
    colnames = [var.name for var in model_variables]

    def cplex_var_types(var):
        if var.cat == constants.LpInteger:
            return 'I'
        else:
            return 'C'
    ctype = [cplex_var_types(var) for var in model_variables]
    ctype = ''.join(ctype)
    lp.solverModel.variables.add(obj=obj, lb=lb, ub=ub, types=ctype, names=colnames)
    rows = []
    senses = []
    rhs = []
    rownames = []
    for name, constraint in lp.constraints.items():
        expr = [(var.name, float(coeff)) for var, coeff in constraint.items()]
        if not expr:
            rows.append(([], []))
        else:
            rows.append(list(zip(*expr)))
        if constraint.sense == constants.LpConstraintLE:
            senses.append('L')
        elif constraint.sense == constants.LpConstraintGE:
            senses.append('G')
        elif constraint.sense == constants.LpConstraintEQ:
            senses.append('E')
        else:
            raise PulpSolverError('Detected an invalid constraint type')
        rownames.append(name)
        rhs.append(float(-constraint.constant))
    lp.solverModel.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=rownames)
    log.debug('set the type of the problem')
    if not self.mip:
        self.solverModel.set_problem_type(cplex.Cplex.problem_type.LP)
    log.debug('set the logging')
    if not self.msg:
        self.setlogfile(None)
    logPath = self.optionsDict.get('logPath')
    if logPath is not None:
        if self.msg:
            warnings.warn('`logPath` argument replaces `msg=1`. The output will be redirected to the log file.')
        self.setlogfile(open(logPath, 'w'))
    gapRel = self.optionsDict.get('gapRel')
    if gapRel is not None:
        self.changeEpgap(gapRel)
    if self.timeLimit is not None:
        self.setTimeLimit(self.timeLimit)
    self.setThreads(self.optionsDict.get('threads', None))
    if self.optionsDict.get('warmStart', False):
        effort = self.solverModel.MIP_starts.effort_level.auto
        start = [(k, v.value()) for k, v in self.n2v.items() if v.value() is not None]
        if not start:
            warnings.warn('No variable with value found: mipStart aborted')
            return
        ind, val = zip(*start)
        self.solverModel.MIP_starts.add(cplex.SparsePair(ind=ind, val=val), effort, '1')