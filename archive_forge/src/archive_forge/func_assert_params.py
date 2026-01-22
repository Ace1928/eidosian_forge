import inspect
import re
import six
def assert_params(self, method, pparams, kparams):
    """
        Checks that positional and keyword parameters match a method
        definition, or raise an ExecutionError.
        @param method: The method to check call signature against.
        @type method: method
        @param pparams: The positional parameters.
        @type pparams: list
        @param kparams: The keyword parameters.
        @type kparams: dict
        @raise ExecutionError: When the check fails.
        """
    spec = inspect.getfullargspec(method)
    args = spec.args[1:]
    pp = spec.varargs
    kw = spec.varkw
    if spec.defaults is None:
        nb_opt_params = 0
    else:
        nb_opt_params = len(spec.defaults)
    nb_max_params = len(args)
    nb_min_params = nb_max_params - nb_opt_params
    req_params = args[:nb_min_params]
    opt_params = args[nb_min_params:]
    unexpected_keywords = sorted(set(kparams) - set(args))
    missing_params = sorted(set(args[len(pparams):]) - set(opt_params) - set(kparams.keys()))
    nb_params = len(pparams) + len(kparams)
    nb_standard_params = len(pparams) + len([param for param in kparams if param in args])
    nb_extended_params = nb_params - nb_standard_params
    self.shell.log.debug('Min params: %d' % nb_min_params)
    self.shell.log.debug('Max params: %d' % nb_max_params)
    self.shell.log.debug('Required params: %s' % ', '.join(req_params))
    self.shell.log.debug('Optional params: %s' % ', '.join(opt_params))
    self.shell.log.debug('Got %s standard params.' % nb_standard_params)
    self.shell.log.debug('Got %s extended params.' % nb_extended_params)
    self.shell.log.debug('Variable positional params: %s' % pp)
    self.shell.log.debug('Variable keyword params: %s' % kw)
    if len(missing_params) == 1:
        raise ExecutionError('Missing required parameter %s' % missing_params[0])
    elif missing_params:
        raise ExecutionError('Missing required parameters %s' % ', '.join(("'%s'" % missing for missing in missing_params)))
    if kw is None:
        if len(unexpected_keywords) == 1:
            raise ExecutionError("Unexpected keyword parameter '%s'." % unexpected_keywords[0])
        elif unexpected_keywords:
            raise ExecutionError('Unexpected keyword parameters %s.' % ', '.join(("'%s'" % kw for kw in unexpected_keywords)))
    all_params = args[:len(pparams)]
    all_params.extend(kparams.keys())
    for param in all_params:
        if all_params.count(param) > 1:
            raise ExecutionError('Duplicate parameter %s.' % param)
    if nb_opt_params == 0 and nb_standard_params != nb_min_params and (pp is None):
        raise ExecutionError('Got %d positionnal parameters, expected exactly %d.' % (nb_standard_params, nb_min_params))
    if nb_standard_params > nb_max_params and pp is None:
        raise ExecutionError('Got %d positionnal parameters, expected at most %d.' % (nb_standard_params, nb_max_params))