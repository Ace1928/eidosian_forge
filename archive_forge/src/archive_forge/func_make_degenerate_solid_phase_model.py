import pyomo.environ as pyo
import pyomo.dae as dae
def make_degenerate_solid_phase_model():
    """
    From the solid phase thermo package of a moving bed chemical looping
    combustion reactor. This example was first presented in [1]

    [1] Parker, R. Nonlinear programming strategies for dynamic models of
    chemical looping combustion reactors. Pres. AIChE Annual Meeting, 2020.

    """
    m = pyo.ConcreteModel()
    m.components = pyo.Set(initialize=[1, 2, 3])
    m.x = pyo.Var(m.components, initialize=1 / 3)
    m.flow_comp = pyo.Var(m.components, initialize=10)
    m.flow = pyo.Var(initialize=30)
    m.rho = pyo.Var(initialize=1)
    m.sum_eqn = pyo.Constraint(expr=sum((m.x[j] for j in m.components)) - 1 == 0)
    m.holdup_eqn = pyo.Constraint(m.components, expr={j: m.x[j] * m.rho - 1 == 0 for j in m.components})
    m.density_eqn = pyo.Constraint(expr=1 / m.rho - sum((1 / m.x[j] for j in m.components)) == 0)
    m.flow_eqn = pyo.Constraint(m.components, expr={j: m.x[j] * m.flow - m.flow_comp[j] == 0 for j in m.components})
    return m