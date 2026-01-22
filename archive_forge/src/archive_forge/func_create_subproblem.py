import os
import sys
import time
from pyomo.common.dependencies import mpi4py
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo
def create_subproblem(root, farmer, scenario):
    m = pyo.ConcreteModel()
    m.crops = pyo.Set(initialize=farmer.crops, ordered=True)
    m.devoted_acreage = pyo.Var(m.crops)
    m.QuantitySubQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
    m.QuantitySuperQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
    m.QuantityPurchased = pyo.Var(m.crops, bounds=(0.0, None))

    def EnforceCattleFeedRequirement_rule(m, i):
        return farmer.CattleFeedRequirement[i] <= farmer.crop_yield[scenario][i] * m.devoted_acreage[i] + m.QuantityPurchased[i] - m.QuantitySubQuotaSold[i] - m.QuantitySuperQuotaSold[i]
    m.EnforceCattleFeedRequirement = pyo.Constraint(m.crops, rule=EnforceCattleFeedRequirement_rule)

    def LimitAmountSold_rule(m, i):
        return m.QuantitySubQuotaSold[i] + m.QuantitySuperQuotaSold[i] - farmer.crop_yield[scenario][i] * m.devoted_acreage[i] <= 0.0
    m.LimitAmountSold = pyo.Constraint(m.crops, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(m, i):
        return (0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i])
    m.EnforceQuotas = pyo.Constraint(m.crops, rule=EnforceQuotas_rule)
    obj_expr = sum((farmer.PurchasePrice[crop] * m.QuantityPurchased[crop] for crop in m.crops))
    obj_expr -= sum((farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop] for crop in m.crops))
    obj_expr -= sum((farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop] for crop in m.crops))
    m.obj = pyo.Objective(expr=farmer.scenario_probabilities[scenario] * obj_expr)
    complicating_vars_map = pyo.ComponentMap()
    for crop in m.crops:
        complicating_vars_map[root.devoted_acreage[crop]] = m.devoted_acreage[crop]
    return (m, complicating_vars_map)