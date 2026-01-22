from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import binary_type, PY3
from ansible.module_utils.six.moves.http_client import responses as http_responses
def format_error_problem(problem, subproblem_prefix=''):
    error_type = problem.get('type', 'about:blank')
    if 'title' in problem:
        msg = 'Error "{title}" ({type})'.format(type=error_type, title=problem['title'])
    else:
        msg = 'Error {type}'.format(type=error_type)
    if 'detail' in problem:
        msg += ': "{detail}"'.format(detail=problem['detail'])
    subproblems = problem.get('subproblems')
    if subproblems is not None:
        msg = '{msg} Subproblems:'.format(msg=msg)
        for index, problem in enumerate(subproblems):
            index_str = '{prefix}{index}'.format(prefix=subproblem_prefix, index=index)
            msg = '{msg}\n({index}) {problem}'.format(msg=msg, index=index_str, problem=format_error_problem(problem, subproblem_prefix='{0}.'.format(index_str)))
    return msg