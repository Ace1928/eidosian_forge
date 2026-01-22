from contextlib import nullcontext
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.repn import generate_standard_repn
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.repn.plugins.nl_writer import AMPLRepn
from pyomo.contrib.incidence_analysis.config import (
def get_incident_variables(expr, **kwds):
    """Get variables that participate in an expression

    The exact variables returned depends on the method used to determine incidence.
    For example, ``method=IncidenceMethod.identify_variables`` will return all
    variables participating in the expression, while
    ``method=IncidenceMethod.standard_repn`` will return only the variables
    identified by ``generate_standard_repn`` which ignores variables that only
    appear multiplied by a constant factor of zero.

    Keyword arguments must be valid options for ``IncidenceConfig``.

    Parameters
    ----------
    expr: ``NumericExpression``
        Expression to search for variables

    Returns
    -------
    list of VarData
        List containing the variables that participate in the expression

    Example
    -------

    .. doctest::

       >>> import pyomo.environ as pyo
       >>> from pyomo.contrib.incidence_analysis import get_incident_variables
       >>> m = pyo.ConcreteModel()
       >>> m.x = pyo.Var([1, 2, 3])
       >>> expr = m.x[1] + 2*m.x[2] + 3*m.x[3]**2
       >>> print([v.name for v in get_incident_variables(expr)])
       ['x[1]', 'x[2]', 'x[3]']
       >>> print([v.name for v in get_incident_variables(expr, linear_only=True)])
       ['x[1]', 'x[2]']

    """
    config = get_config_from_kwds(**kwds)
    method = config.method
    include_fixed = config.include_fixed
    linear_only = config.linear_only
    amplrepnvisitor = config._ampl_repn_visitor
    if linear_only and method is IncidenceMethod.identify_variables:
        raise RuntimeError('linear_only=True is not supported when using identify_variables')
    if include_fixed and method is IncidenceMethod.ampl_repn:
        raise RuntimeError('include_fixed=True is not supported when using ampl_repn')
    if method is IncidenceMethod.ampl_repn and amplrepnvisitor is None:
        raise RuntimeError('_ampl_repn_visitor must be provided when using ampl_repn')
    if method is IncidenceMethod.identify_variables:
        return _get_incident_via_identify_variables(expr, include_fixed)
    elif method is IncidenceMethod.standard_repn:
        return _get_incident_via_standard_repn(expr, include_fixed, linear_only, compute_values=False)
    elif method is IncidenceMethod.standard_repn_compute_values:
        return _get_incident_via_standard_repn(expr, include_fixed, linear_only, compute_values=True)
    elif method is IncidenceMethod.ampl_repn:
        return _get_incident_via_ampl_repn(expr, linear_only, amplrepnvisitor)
    else:
        raise ValueError(f'Unrecognized value {method} for the method used to identify incident variables. See the IncidenceMethod enum for valid methods.')