import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.examples.external_grey_box.react_example.reactor_model_residuals import (
def maximize_cb_ratio_residuals_with_output_scaling(show_solver_log=False, additional_options={}):
    m = pyo.ConcreteModel()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.reactor = ExternalGreyBoxBlock(external_model=ReactorModelScaled())
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)
    m.scaling_factor[m.cafcon] = 42.0
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb_ratio'], sense=pyo.maximize)
    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    for k, v in additional_options.items():
        solver.config.options[k] = v
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m