import os
def int_val_fn(v):
    try:
        int(v)
        return True
    except:
        return False