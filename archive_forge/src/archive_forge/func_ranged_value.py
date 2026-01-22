from pyparsing import Word, nums, Suppress, Optional
from datetime import datetime
def ranged_value(expr, minval=None, maxval=None):
    if minval is None and maxval is None:
        raise ValueError('minval or maxval must be specified')
    inRangeCondition = {(True, False): lambda s, l, t: t[0] <= maxval, (False, True): lambda s, l, t: minval <= t[0], (False, False): lambda s, l, t: minval <= t[0] <= maxval}[minval is None, maxval is None]
    outOfRangeMessage = {(True, False): 'value is greater than %s' % maxval, (False, True): 'value is less than %s' % minval, (False, False): 'value is not in the range ({0} to {1})'.format(minval, maxval)}[minval is None, maxval is None]
    return expr().addCondition(inRangeCondition, message=outOfRangeMessage)