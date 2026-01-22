from pyomo.environ import (
import pyomo.contrib.parmest.parmest as parmest
def simple_reaction_model(data):
    model = ConcreteModel()
    model.x1 = Param(initialize=float(data['x1']))
    model.x2 = Param(initialize=float(data['x2']))
    model.rxn = RangeSet(2)
    initial_guess = {1: 750, 2: 1200}
    model.k = Var(model.rxn, initialize=initial_guess, within=PositiveReals)
    model.y = Expression(expr=exp(-model.k[1] * model.x1 * exp(-model.k[2] / model.x2)))
    model.k.fix()

    def ComputeFirstStageCost_rule(model):
        return 0
    model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

    def AllMeasurements(m):
        return (float(data['y']) - m.y) ** 2
    model.SecondStageCost = Expression(rule=AllMeasurements)

    def total_cost_rule(m):
        return m.FirstStageCost + m.SecondStageCost
    model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)
    return model