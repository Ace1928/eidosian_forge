import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.model import (
def run_cstr_openloop(inputs, model_horizon=1.0, ntfe=10, simulation_steps=15, tee=False):
    m = create_instance(horizon=model_horizon, ntfe=ntfe)
    dynamic_interface = mpc.DynamicModelInterface(m, m.time)
    sim_t0 = 0.0
    sim_data = dynamic_interface.get_data_at_time([sim_t0])
    solver = pyo.SolverFactory('ipopt')
    non_initial_model_time = list(m.time)[1:]
    for i in range(simulation_steps):
        sim_t0 = i * model_horizon
        sim_time = [sim_t0 + t for t in m.time]
        new_inputs = mpc.data.convert.interval_to_series(inputs, time_points=sim_time)
        new_inputs.shift_time_points(m.time.first() - sim_t0)
        dynamic_interface.load_data(new_inputs, tolerance=1e-06)
        res = solver.solve(m, tee=tee)
        pyo.assert_optimal_termination(res)
        m_data = dynamic_interface.get_data_at_time(non_initial_model_time)
        m_data.shift_time_points(sim_t0 - m.time.first())
        sim_data.concatenate(m_data)
        tf_data = dynamic_interface.get_data_at_time(m.time.last())
        dynamic_interface.load_data(tf_data)
    return (m, sim_data)