import collections
from ._caveat import error_caveat
from ._utils import condition_with_prefix
def resolve_caveat(self, cav):
    """ Resolves the given caveat(string) by using resolve to map from its
        schema namespace to the appropriate prefix.
        If there is no registered prefix for the namespace, it returns an error
        caveat.
        If cav.namespace is empty or cav.location is non-empty, it returns cav
        unchanged.

        It does not mutate ns and may be called concurrently with other
        non-mutating Namespace methods.
        :return: Caveat object
        """
    if cav.namespace == '' or cav.location != '':
        return cav
    prefix = self.resolve(cav.namespace)
    if prefix is None:
        err_cav = error_caveat('caveat {} in unregistered namespace {}'.format(cav.condition, cav.namespace))
        if err_cav.namespace != cav.namespace:
            prefix = self.resolve(err_cav.namespace)
            if prefix is None:
                prefix = ''
        cav = err_cav
    if prefix != '':
        cav.condition = condition_with_prefix(prefix, cav.condition)
    cav.namespace = ''
    return cav