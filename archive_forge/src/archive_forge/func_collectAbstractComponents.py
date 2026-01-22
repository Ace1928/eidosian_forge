from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param
def collectAbstractComponents(model):
    """
    Returns all abstract constraints, objectives, variables,
    parameters, and sets. Does not query instance values, if
    present. Returns nested dictionaries describing the model,
    including all rules, domains, bounds, and index sets.
    """
    cp = model.clone()
    constraints = {}
    variables = {}
    objectives = {}
    parameters = {}
    sets = {}
    conName = 'Constraint'
    varName = 'Var'
    objName = 'Objective'
    paramName = 'Param'
    setName = 'Set'
    index = 'index'
    bounds = 'bounds'
    domain = 'domain'
    initialize = 'initialize'
    rule = 'rule'
    for comp in cp._ctypes:
        if issubclass(comp, Constraint):
            comps = cp.component_map(comp, active=True)
            for name, obj in [(name, comps[name]) for name in comps]:
                data = {}
                data[index] = _getAbstractIndices(obj)
                data[rule] = _getAbstractRule(obj)
                constraints[name] = data
        if issubclass(comp, Objective):
            comps = cp.component_map(comp, active=True)
            for name, obj in [(name, comps[name]) for name in comps]:
                data = {}
                data[index] = _getAbstractIndices(obj)
                data[rule] = _getAbstractRule(obj)
                objectives[name] = data
        if issubclass(comp, Var):
            comps = cp.component_map(comp, active=True)
            for name, obj in [(name, comps[name]) for name in comps]:
                data = {}
                data[index] = _getAbstractIndices(obj)
                data[domain] = _getAbstractDomain(obj)
                data[bounds] = _getAbstractBounds(obj)
                variables[name] = data
        if issubclass(comp, Set):
            comps = cp.component_map(comp, active=True)
            for name, obj in [(name, comps[name]) for name in comps]:
                data = {}
                data[index] = _getAbstractIndices(obj)
                data[domain] = _getAbstractDomain(obj)
                sets[name] = data
        if issubclass(comp, Param):
            comps = cp.component_map(comp, active=True)
            for name, obj in [(name, comps[name]) for name in comps]:
                data = {}
                data[index] = _getAbstractIndices(obj)
                data[domain] = _getAbstractDomain(obj)
                parameters[name] = data
    master = {}
    master[conName] = constraints
    master[objName] = objectives
    master[varName] = variables
    master[paramName] = parameters
    master[setName] = sets
    return master