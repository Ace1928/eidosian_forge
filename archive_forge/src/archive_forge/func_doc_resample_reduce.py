from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string
def doc_resample_reduce(result, refer_to, params=None, compatibility_params=True):
    """
    Build decorator which adds docstring for the resample reduce method.

    Parameters
    ----------
    result : str
        The result of the method.
    refer_to : str
        Method name in ``modin.pandas.resample.Resampler`` module to refer to for
        more information about parameters and output format.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.
    compatibility_params : bool, default: True
        Whether method takes `*args` and `**kwargs` that do not affect
        the result.

    Returns
    -------
    callable
    """
    action = f'compute {result} for each group'
    params_substitution = '\n        *args : iterable\n            Serves the compatibility purpose. Does not affect the result.\n        **kwargs : dict\n            Serves the compatibility purpose. Does not affect the result.\n        ' if compatibility_params else ''
    if params:
        params_substitution = format_string('{params}\n{params_substitution}', params=params, params_substitution=params_substitution)
    build_rules = f'\n            - Labels on the specified axis are the group names (time-stamps)\n            - Labels on the opposite of specified axis are preserved.\n            - Each element of QueryCompiler is the {result} for the\n              corresponding group and column/row.'
    return doc_resample(action=action, extra_params=params_substitution, build_rules=build_rules, refer_to=refer_to)