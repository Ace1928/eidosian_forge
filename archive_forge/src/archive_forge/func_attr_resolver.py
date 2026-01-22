def attr_resolver(attname, default_value, root, info, **args):
    return getattr(root, attname, default_value)