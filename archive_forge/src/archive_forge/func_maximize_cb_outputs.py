import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.examples.external_grey_box.react_example.reactor_model_outputs import (
def maximize_cb_outputs(show_solver_log=False):
    m = pyo.ConcreteModel()
    m.reactor = ExternalGreyBoxBlock(external_model=ReactorConcentrationsOutputModel())
    m.k1con = pyo.Constraint(expr=m.reactor.inputs['k1'] == 5 / 6)
    m.k2con = pyo.Constraint(expr=m.reactor.inputs['k2'] == 5 / 3)
    m.k3con = pyo.Constraint(expr=m.reactor.inputs['k3'] == 1 / 6000)
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb'], sense=pyo.maximize)
    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m