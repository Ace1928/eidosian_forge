def register_context(name, cls, *args, **kwargs):
    """Register a new context.
    """
    instance = cls(*args, **kwargs)
    proxy = ProxyContext(instance)
    _contexts[name] = {'cls': cls, 'args': args, 'kwargs': kwargs, 'proxy': proxy}
    _default_context[name] = instance
    return proxy