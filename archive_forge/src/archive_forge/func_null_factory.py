import dns._features
import dns.asyncbackend
def null_factory(*args, **kwargs):
    return NullContext(None)