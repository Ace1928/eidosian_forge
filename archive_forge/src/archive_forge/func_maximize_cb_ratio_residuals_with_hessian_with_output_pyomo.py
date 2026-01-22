import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.examples.external_grey_box.react_example.reactor_model_residuals import (
def maximize_cb_ratio_residuals_with_hessian_with_output_pyomo(show_solver_log=False, additional_options={}):
    m = create_pyomo_reactor_model()
    solver = pyo.SolverFactory('ipopt')
    for k, v in additional_options.items():
        solver.options[k] = v
    solver.options['linear_solver'] = 'mumps'
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m