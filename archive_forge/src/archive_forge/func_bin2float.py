from functools import wraps
def bin2float(min_, max_, nbits):
    """Convert a binary array into an array of float where each
    float is composed of *nbits* and is between *min_* and *max_*
    and return the result of the decorated function.

    """

    def wrap(function):

        @wraps(function)
        def wrapped_function(individual, *args, **kargs):
            nelem = len(individual) // nbits
            decoded = [0] * nelem
            for i in range(nelem):
                gene = int(''.join(map(str, individual[i * nbits:i * nbits + nbits])), 2)
                div = 2 ** nbits - 1
                temp = gene / div
                decoded[i] = min_ + temp * (max_ - min_)
            return function(decoded, *args, **kargs)
        return wrapped_function
    return wrap