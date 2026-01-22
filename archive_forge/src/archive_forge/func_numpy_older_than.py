from packaging.version import parse as version
def numpy_older_than(ver: str) -> bool:
    """Returns True if the numpy version is older than the given version."""
    import numpy
    return version(numpy.__version__) < version(ver)