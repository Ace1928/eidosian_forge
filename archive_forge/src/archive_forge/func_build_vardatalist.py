import pyomo.environ as pyo
def build_vardatalist(self, model, varlist=None):
    """
    Convert a list of pyomo variables to a list of ScalarVar and _GeneralVarData. If varlist is none, builds a
    list of all variables in the model. The new list is stored in the vars_to_tighten attribute. By CD Laird

    Parameters
    ----------
    model: ConcreteModel
    varlist: None or list of pyo.Var
    """
    vardatalist = None
    if varlist is None:
        raise RuntimeError('varlist is None in scenario_tree.build_vardatalist')
        vardatalist = [v for v in model.component_data_objects(pyo.Var, active=True, sort=True)]
    elif isinstance(varlist, pyo.Var):
        varlist = [varlist]
    if vardatalist is None:
        vardatalist = list()
        for v in varlist:
            if v.is_indexed():
                vardatalist.extend([v[i] for i in sorted(v.keys())])
            else:
                vardatalist.append(v)
    return vardatalist