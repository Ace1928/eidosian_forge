def register_internal_serializable(path, symbol):
    global REGISTERED_NAMES_TO_OBJS
    if isinstance(path, (list, tuple)):
        name = path[0]
    else:
        name = path
    REGISTERED_NAMES_TO_OBJS[name] = symbol
    REGISTERED_OBJS_TO_NAMES[symbol] = name