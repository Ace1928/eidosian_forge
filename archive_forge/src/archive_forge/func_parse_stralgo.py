import datetime
from redis.utils import str_if_bytes
def parse_stralgo(response, **options):
    """
    Parse the response from `STRALGO` command.
    Without modifiers the returned value is string.
    When LEN is given the command returns the length of the result
    (i.e integer).
    When IDX is given the command returns a dictionary with the LCS
    length and all the ranges in both the strings, start and end
    offset for each string, where there are matches.
    When WITHMATCHLEN is given, each array representing a match will
    also have the length of the match at the beginning of the array.
    """
    if options.get('len', False):
        return int(response)
    if options.get('idx', False):
        if options.get('withmatchlen', False):
            matches = [[int(match[-1])] + list(map(tuple, match[:-1])) for match in response[1]]
        else:
            matches = [list(map(tuple, match)) for match in response[1]]
        return {str_if_bytes(response[0]): matches, str_if_bytes(response[2]): int(response[3])}
    return str_if_bytes(response)