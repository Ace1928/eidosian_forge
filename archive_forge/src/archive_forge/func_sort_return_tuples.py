import datetime
from redis.utils import str_if_bytes
def sort_return_tuples(response, **options):
    """
    If ``groups`` is specified, return the response as a list of
    n-element tuples with n being the value found in options['groups']
    """
    if not response or not options.get('groups'):
        return response
    n = options['groups']
    return list(zip(*[response[i::n] for i in range(n)]))