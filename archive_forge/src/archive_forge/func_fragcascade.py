import sys
def fragcascade(attr, seq_type, doc=''):
    """Return a getter property with cascading setter, for HSPFragment objects.

    Similar to ``partialcascade``, but for HSPFragment objects and acts on ``query``
    or ``hit`` properties of the object if they are not None.

    """
    assert seq_type in ('hit', 'query')
    attr_name = f'_{seq_type}_{attr}'

    def getter(self):
        return getattr(self, attr_name)

    def setter(self, value):
        setattr(self, attr_name, value)
        seq = getattr(self, seq_type)
        if seq is not None:
            setattr(seq, attr, value)
    return property(fget=getter, fset=setter, doc=doc)