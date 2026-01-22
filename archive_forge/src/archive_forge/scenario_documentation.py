import pickle
from enum import Enum
from collections import namedtuple

        Generate scenario data for the given parameter dictionary.

        Returns:
        -------
        ScenarioData: a namedtuple containing scenarios information.
        ScenarioData.scenario: a list of dictionaries, each dictionary contains a perturbed scenario
        ScenarioData.scena_num: a dict of scenario number related to one parameter
        ScenarioData.eps_abs: keys are parameter name, values are the step it is perturbed
        ScenarioData.scenario_indices: a list of scenario indices


        For e.g., if a dict {'P':100, 'D':20} is given, step=0.1, formula='central', it will return:
            self.ScenarioData.scenario: [{'P':101, 'D':20}, {'P':99, 'D':20}, {'P':100, 'D':20.2}, {'P':100, 'D':19.8}],
            self.ScenarioData.scena_num: {'P':[0,1], 'D':[2,3]}}
            self.ScenarioData.eps_abs: {'P': 2.0, 'D': 0.4}
            self.ScenarioData.scenario_indices: [0,1,2,3]
        if formula ='forward', it will return:
            self.ScenarioData.scenario:[{'P':101, 'D':20}, {'P':100, 'D':20.2}, {'P':100, 'D':20}],
            self.ScenarioData.scena_num: {'P':[0,2], 'D':[1,2]}}
            self.ScenarioData.eps_abs: {'P': 2.0, 'D': 0.4}
            self.ScenarioData.scenario_indices: [0,1,2]
        