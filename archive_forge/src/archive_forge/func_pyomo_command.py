import logging
def pyomo_command(name=None, doc=None):

    def wrap(fn):
        if name is None:
            logger.error('Error applying decorator.  No command name!')
            return
        if doc is None:
            logger.error('Error applying decorator.  No command documentation!')
            return
        global registry
        registry[name] = doc
        return fn
    return wrap