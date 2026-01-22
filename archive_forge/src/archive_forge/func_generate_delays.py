from collections import abc
import functools
import itertools
def generate_delays(delay, max_delay, multiplier=2):
    """Generator/iterator that provides back delays values.

    The values it generates increments by a given multiple after each
    iteration (using the max delay as a upper bound). Negative values
    will never be generated... and it will iterate forever (ie it will never
    stop generating values).
    """
    if max_delay < 0:
        raise ValueError('Provided delay (max) must be greater than or equal to zero')
    if delay < 0:
        raise ValueError('Provided delay must start off greater than or equal to zero')
    if multiplier < 1.0:
        raise ValueError('Provided multiplier must be greater than or equal to 1.0')

    def _gen_it():
        curr_delay = delay
        while True:
            curr_delay = max(0, min(max_delay, curr_delay))
            yield curr_delay
            curr_delay = curr_delay * multiplier
    return _gen_it()