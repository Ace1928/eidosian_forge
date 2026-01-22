from ._swatches import _swatches
def swatches(template=None):
    return _swatches(__name__, globals(), template)