from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string
def doc_resample_fillna(method, refer_to, params=None, overwrite_template_params=False):
    """
    Build decorator which adds docstring for the resample fillna query compiler method.

    Parameters
    ----------
    method : str
        Fillna method name.
    refer_to : str
        Method name in ``modin.pandas.resample.Resampler`` module to refer to for
        more information about parameters and output format.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.
    overwrite_template_params : bool, default: False
        If `params` is specified indicates whether to overwrite method parameters in
        the docstring template or append then at the end.

    Returns
    -------
    callable
    """
    action = f'fill missing values in each group independently using {method} method'
    params_substitution = 'limit : int\n'
    if params:
        params_substitution = params if overwrite_template_params else format_string('{params}\n{params_substitution}', params=params, params_substitution=params_substitution)
    build_rules = '- QueryCompiler contains unsampled data with missing values filled.'
    return doc_resample(action=action, extra_params=params_substitution, build_rules=build_rules, refer_to=refer_to)