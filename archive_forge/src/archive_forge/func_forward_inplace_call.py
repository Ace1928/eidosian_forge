from functools import update_wrapper
import numpy as np
def forward_inplace_call(name):
    arraymeth = getattr(np.ndarray, name)

    def f(self, obj):
        a = self.__array__()
        arraymeth(a, obj)
        return self
    update_wrapper(f, arraymeth)
    return f