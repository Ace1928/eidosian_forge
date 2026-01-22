from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string
def doc_resample_agg(action, output, refer_to, params=None):
    """
    Build decorator which adds docstring for the resample aggregation method.

    Parameters
    ----------
    action : str
        What method does with the resampled data.
    output : str
        What is the content of column names in the result.
    refer_to : str
        Method name in ``modin.pandas.resample.Resampler`` module to refer to for
        more information about parameters and output format.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.

    Returns
    -------
    callable
    """
    action = f'{action} for each group over the specified axis'
    params_substitution = '\n        *args : iterable\n            Positional arguments to pass to the aggregation function.\n        **kwargs : dict\n            Keyword arguments to pass to the aggregation function.\n        '
    if params:
        params_substitution = format_string('{params}\n{params_substitution}', params=params, params_substitution=params_substitution)
    build_rules = f'\n            - Labels on the specified axis are the group names (time-stamps)\n            - Labels on the opposite of specified axis are a MultiIndex, where first level\n              contains preserved labels of this axis and the second level is the {output}.\n            - Each element of QueryCompiler is the result of corresponding function for the\n              corresponding group and column/row.'
    return doc_resample(action=action, extra_params=params_substitution, build_rules=build_rules, refer_to=refer_to)