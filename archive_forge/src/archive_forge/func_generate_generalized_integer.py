import codecs
def generate_generalized_integer(N, bias):
    """3.3 Generalized variable-length integers"""
    result = bytearray()
    j = 0
    while 1:
        t = T(j, bias)
        if N < t:
            result.append(digits[N])
            return bytes(result)
        result.append(digits[t + (N - t) % (36 - t)])
        N = (N - t) // (36 - t)
        j += 1