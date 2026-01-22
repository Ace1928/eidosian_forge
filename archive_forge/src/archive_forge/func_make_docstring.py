import inspect
from textwrap import TextWrapper
def make_docstring(fn, override_dict=None, append_dict=None):
    override_dict = {} if override_dict is None else override_dict
    append_dict = {} if append_dict is None else append_dict
    tw = TextWrapper(width=75, initial_indent='    ', subsequent_indent='    ')
    result = (fn.__doc__ or '') + '\nParameters\n----------\n'
    for param in getfullargspec(fn)[0]:
        if override_dict.get(param):
            param_doc = list(override_dict[param])
        else:
            param_doc = list(docs[param])
            if append_dict.get(param):
                param_doc += append_dict[param]
        param_desc_list = param_doc[1:]
        param_desc = tw.fill(' '.join(param_desc_list or '')) if param in docs or param in override_dict else '(documentation missing from map)'
        param_type = param_doc[0]
        result += '%s: %s\n%s\n' % (param, param_type, param_desc)
    result += '\nReturns\n-------\n'
    result += '    plotly.graph_objects.Figure'
    return result