from .python3_compat import iterkeys, iteritems, Mapping  #, u
def munchify(x, factory=Munch):
    """ Recursively transforms a dictionary into a Munch via copy.

        >>> b = munchify({'urmom': {'sez': {'what': 'what'}}})
        >>> b.urmom.sez.what
        'what'

        munchify can handle intermediary dicts, lists and tuples (as well as
        their subclasses), but ymmv on custom datatypes.

        >>> b = munchify({ 'lol': ('cats', {'hah':'i win again'}),
        ...         'hello': [{'french':'salut', 'german':'hallo'}] })
        >>> b.hello[0].french
        'salut'
        >>> b.lol[1].hah
        'i win again'

        nb. As dicts are not hashable, they cannot be nested in sets/frozensets.
    """
    seen = dict()

    def munchify_cycles(obj):
        try:
            return seen[id(obj)]
        except KeyError:
            pass
        seen[id(obj)] = partial = pre_munchify(obj)
        return post_munchify(partial, obj)

    def pre_munchify(obj):
        if isinstance(obj, Mapping):
            return factory({})
        elif isinstance(obj, list):
            return type(obj)()
        elif isinstance(obj, tuple):
            type_factory = getattr(obj, '_make', type(obj))
            return type_factory((munchify_cycles(item) for item in obj))
        else:
            return obj

    def post_munchify(partial, obj):
        if isinstance(obj, Mapping):
            partial.update(((k, munchify_cycles(obj[k])) for k in iterkeys(obj)))
        elif isinstance(obj, list):
            partial.extend((munchify_cycles(item) for item in obj))
        elif isinstance(obj, tuple):
            for item_partial, item in zip(partial, obj):
                post_munchify(item_partial, item)
        return partial
    return munchify_cycles(x)