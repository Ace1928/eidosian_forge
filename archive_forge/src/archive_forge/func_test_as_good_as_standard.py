import logging
from itertools import product
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
def test_as_good_as_standard(self):
    standard_model = build_model()
    SolverFactory('ipopt').solve(standard_model)
    standard_objective_value = value(next(standard_model.component_data_objects(Objective, active=True)))
    fresh_model = build_model()
    multistart_iterations = 10
    test_trials = 10
    for strategy, _ in product(strategies.keys(), range(test_trials)):
        m2 = fresh_model.clone()
        SolverFactory('multistart').solve(m2, iterations=multistart_iterations, strategy=strategy)
        clone_objective_value = value(next(m2.component_data_objects(Objective, active=True)))
        self.assertGreaterEqual(clone_objective_value, standard_objective_value)