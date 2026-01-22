import datetime
from redis.utils import str_if_bytes
def parse_geosearch_generic(response, **options):
    """
    Parse the response of 'GEOSEARCH', GEORADIUS' and 'GEORADIUSBYMEMBER'
    commands according to 'withdist', 'withhash' and 'withcoord' labels.
    """
    try:
        if options['store'] or options['store_dist']:
            return response
    except KeyError:
        return response
    if type(response) != list:
        response_list = [response]
    else:
        response_list = response
    if not options['withdist'] and (not options['withcoord']) and (not options['withhash']):
        return response_list
    cast = {'withdist': float, 'withcoord': lambda ll: (float(ll[0]), float(ll[1])), 'withhash': int}
    f = [lambda x: x]
    f += [cast[o] for o in ['withdist', 'withhash', 'withcoord'] if options[o]]
    return [list(map(lambda fv: fv[0](fv[1]), zip(f, r))) for r in response_list]