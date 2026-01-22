def pickle_union(obj):
    import functools, operator
    return (functools.reduce, (operator.or_, obj.__args__))