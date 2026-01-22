import time
def list2cols(cols, objs):
    return (cols, [tuple([o[k] for k in cols]) for o in objs])