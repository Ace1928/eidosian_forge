from collections.abc import Mapping
def fromYAML(cls, stream, *args, **kwargs):
    factory = lambda d: cls(*args + (d,), **kwargs)
    loader_class = kwargs.pop('Loader', yaml.FullLoader)
    return munchify(yaml.load(stream, Loader=loader_class), factory=factory)