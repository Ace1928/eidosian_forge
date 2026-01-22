import deap
from copy import copy
def set_param_recursive(pipeline_steps, parameter, value):
    """Recursively iterate through all objects in the pipeline and set a given parameter.

    Parameters
    ----------
    pipeline_steps: array-like
        List of (str, obj) tuples from a scikit-learn pipeline or related object
    parameter: str
        The parameter to assign a value for in each pipeline object
    value: any
        The value to assign the parameter to in each pipeline object
    Returns
    -------
    None

    """
    for _, obj in pipeline_steps:
        recursive_attrs = ['steps', 'transformer_list', 'estimators']
        for attr in recursive_attrs:
            if hasattr(obj, attr):
                set_param_recursive(getattr(obj, attr), parameter, value)
        if hasattr(obj, 'estimator'):
            est = getattr(obj, 'estimator')
            if hasattr(est, parameter):
                setattr(est, parameter, value)
        if hasattr(obj, parameter):
            setattr(obj, parameter, value)