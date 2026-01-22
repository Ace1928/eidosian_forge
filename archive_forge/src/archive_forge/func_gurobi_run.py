import re
from gurobipy import gurobi, read, GRB
import sys
def gurobi_run(model_file, warmstart_file, soln_file, mipgap, options, suffixes):
    extract_duals = False
    extract_slacks = False
    extract_reduced_costs = False
    for suffix in suffixes:
        flag = False
        if re.match(suffix, 'dual'):
            extract_duals = True
            flag = True
        if re.match(suffix, 'slack'):
            extract_slacks = True
            flag = True
        if re.match(suffix, 'rc'):
            extract_reduced_costs = True
            flag = True
        if not flag:
            print('***The GUROBI solver plugin cannot extract solution suffix=' + suffix)
            return
    model = read(model_file)
    if GUROBI_VERSION[0] >= 5:
        if extract_reduced_costs is True or extract_duals is True:
            model.setParam(GRB.Param.QCPDual, 1)
    if model is None:
        print('***The GUROBI solver plugin failed to load the input LP file=' + soln_file)
        return
    if warmstart_file is not None:
        model.read(warmstart_file)
    if mipgap is not None:
        model.setParam('MIPGap', mipgap)
    for key, value in options.items():
        try:
            model.setParam(key, value)
        except TypeError:
            if not _is_numeric(value):
                raise
            model.setParam(key, float(value))
    if 'relax_integrality' in options:
        for v in model.getVars():
            if v.vType != GRB.CONTINUOUS:
                v.vType = GRB.CONTINUOUS
        model.update()
    model.optimize()
    wall_time = model.getAttr(GRB.Attr.Runtime)
    solver_status = model.getAttr(GRB.Attr.Status)
    solution_status = None
    return_code = 0
    if solver_status == GRB.LOADED:
        status = 'aborted'
        message = 'Model is loaded, but no solution information is available.'
        term_cond = 'error'
        solution_status = 'unknown'
    elif solver_status == GRB.OPTIMAL:
        status = 'ok'
        message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
        term_cond = 'optimal'
        solution_status = 'optimal'
    elif solver_status == GRB.INFEASIBLE:
        status = 'warning'
        message = 'Model was proven to be infeasible.'
        term_cond = 'infeasible'
        solution_status = 'infeasible'
    elif solver_status == GRB.INF_OR_UNBD:
        status = 'warning'
        message = 'Problem proven to be infeasible or unbounded.'
        term_cond = 'infeasibleOrUnbounded'
        solution_status = 'unsure'
    elif solver_status == GRB.UNBOUNDED:
        status = 'warning'
        message = 'Model was proven to be unbounded.'
        term_cond = 'unbounded'
        solution_status = 'unbounded'
    elif solver_status == GRB.CUTOFF:
        status = 'aborted'
        message = 'Optimal objective for model was proven to be worse than the value specified in the Cutoff  parameter. No solution information is available.'
        term_cond = 'minFunctionValue'
        solution_status = 'unknown'
    elif solver_status == GRB.ITERATION_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter.'
        term_cond = 'maxIterations'
        solution_status = 'stoppedByLimit'
    elif solver_status == GRB.NODE_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.'
        term_cond = 'maxEvaluations'
        solution_status = 'stoppedByLimit'
    elif solver_status == GRB.TIME_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.'
        term_cond = 'maxTimeLimit'
        solution_status = 'stoppedByLimit'
    elif hasattr(GRB, 'WORK_LIMIT') and solver_status == GRB.WORK_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the work expended exceeded the value specified in the WorkLimit parameter.'
        term_cond = 'maxTimeLimit'
        solution_status = 'stoppedByLimit'
    elif solver_status == GRB.SOLUTION_LIMIT:
        status = 'aborted'
        message = 'Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.'
        term_cond = 'stoppedByLimit'
        solution_status = 'stoppedByLimit'
    elif solver_status == GRB.INTERRUPTED:
        status = 'aborted'
        message = 'Optimization was terminated by the user.'
        term_cond = 'error'
        solution_status = 'error'
    elif solver_status == GRB.NUMERIC:
        status = 'error'
        message = 'Optimization was terminated due to unrecoverable numerical difficulties.'
        term_cond = 'error'
        solution_status = 'error'
    elif solver_status == GRB.SUBOPTIMAL:
        status = 'warning'
        message = 'Unable to satisfy optimality tolerances; a sub-optimal solution is available.'
        term_cond = 'other'
        solution_status = 'feasible'
    elif solver_status is not None and solver_status == getattr(GRB, 'USER_OBJ_LIMIT', None):
        status = 'aborted'
        message = 'User specified an objective limit (a bound on either the best objective or the best bound), and that limit has been reached. Solution is available.'
        term_cond = 'other'
        solution_status = 'stoppedByLimit'
    else:
        status = 'error'
        message = 'Unhandled Gurobi solve status (' + str(solver_status) + ')'
        term_cond = 'error'
        solution_status = 'error'
    assert solution_status is not None
    sense = model.getAttr(GRB.Attr.ModelSense)
    try:
        obj_value = model.getAttr(GRB.Attr.ObjVal)
    except:
        obj_value = None
        if term_cond == 'unbounded':
            if sense < 0:
                obj_value = float('inf')
            else:
                obj_value = float('-inf')
        elif term_cond == 'infeasible':
            if sense < 0:
                obj_value = float('-inf')
            else:
                obj_value = float('inf')
    solnfile = open(soln_file, 'w+')
    solnfile.write('section:problem\n')
    name = model.getAttr(GRB.Attr.ModelName)
    solnfile.write('name: ' + name + '\n')
    try:
        bound = model.getAttr(GRB.Attr.ObjBound)
    except Exception:
        if term_cond == 'optimal':
            bound = obj_value
        else:
            bound = None
    if sense < 0:
        solnfile.write('sense:maximize\n')
        if bound is None:
            solnfile.write('upper_bound: %f\n' % float('inf'))
        else:
            solnfile.write('upper_bound: %s\n' % str(bound))
    else:
        solnfile.write('sense:minimize\n')
        if bound is None:
            solnfile.write('lower_bound: %f\n' % float('-inf'))
        else:
            solnfile.write('lower_bound: %s\n' % str(bound))
    n_objs = 1
    solnfile.write('number_of_objectives: %d\n' % n_objs)
    cons = model.getConstrs()
    qcons = []
    if GUROBI_VERSION[0] >= 5:
        qcons = model.getQConstrs()
    solnfile.write('number_of_constraints: %d\n' % (len(cons) + len(qcons) + model.NumSOS,))
    vars = model.getVars()
    solnfile.write('number_of_variables: %d\n' % len(vars))
    n_binvars = model.getAttr(GRB.Attr.NumBinVars)
    solnfile.write('number_of_binary_variables: %d\n' % n_binvars)
    n_intvars = model.getAttr(GRB.Attr.NumIntVars)
    solnfile.write('number_of_integer_variables: %d\n' % n_intvars)
    solnfile.write('number_of_continuous_variables: %d\n' % (len(vars) - n_intvars,))
    solnfile.write('number_of_nonzeros: %d\n' % model.getAttr(GRB.Attr.NumNZs))
    solnfile.write('section:solver\n')
    solnfile.write('status: %s\n' % status)
    solnfile.write('return_code: %s\n' % return_code)
    solnfile.write('message: %s\n' % message)
    solnfile.write('wall_time: %s\n' % str(wall_time))
    solnfile.write('termination_condition: %s\n' % term_cond)
    solnfile.write('termination_message: %s\n' % message)
    is_discrete = False
    if model.getAttr(GRB.Attr.IsMIP):
        is_discrete = True
    if term_cond == 'optimal' or model.getAttr(GRB.Attr.SolCount) >= 1:
        solnfile.write('section:solution\n')
        solnfile.write('status: %s\n' % solution_status)
        solnfile.write('message: %s\n' % message)
        solnfile.write('objective: %s\n' % str(obj_value))
        solnfile.write('gap: 0.0\n')
        vals = model.getAttr('X', vars)
        names = model.getAttr('VarName', vars)
        for val, name in zip(vals, names):
            solnfile.write('var: %s : %s\n' % (str(name), str(val)))
        if is_discrete is False and extract_reduced_costs is True:
            vals = model.getAttr('Rc', vars)
            for val, name in zip(vals, names):
                solnfile.write('varrc: %s : %s\n' % (str(name), str(val)))
        if extract_duals or extract_slacks:
            con_names = model.getAttr('ConstrName', cons)
            if GUROBI_VERSION[0] >= 5:
                qcon_names = model.getAttr('QCName', qcons)
        if is_discrete is False and extract_duals is True:
            vals = model.getAttr('Pi', cons)
            for val, name in zip(vals, con_names):
                solnfile.write('constraintdual: %s : %s\n' % (str(name), str(val)))
            if GUROBI_VERSION[0] >= 5:
                vals = model.getAttr('QCPi', qcons)
                for val, name in zip(vals, qcon_names):
                    solnfile.write('constraintdual: %s : %s\n' % (str(name), str(val)))
        if extract_slacks is True:
            vals = model.getAttr('Slack', cons)
            for val, name in zip(vals, con_names):
                solnfile.write('constraintslack: %s : %s\n' % (str(name), str(val)))
            if GUROBI_VERSION[0] >= 5:
                vals = model.getAttr('QCSlack', qcons)
                for val, name in zip(vals, qcon_names):
                    solnfile.write('constraintslack: %s : %s\n' % (str(name), str(val)))
    solnfile.close()