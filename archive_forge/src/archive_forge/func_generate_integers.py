import codecs
def generate_integers(baselen, deltas):
    """3.4 Bias adaptation"""
    result = bytearray()
    bias = 72
    for points, delta in enumerate(deltas):
        s = generate_generalized_integer(delta, bias)
        result.extend(s)
        bias = adapt(delta, points == 0, baselen + points + 1)
    return bytes(result)