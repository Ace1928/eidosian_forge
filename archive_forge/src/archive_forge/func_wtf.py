import dill
def wtf(x, y, z):

    def zzz():
        return x

    def yyy():
        return y

    def xxx():
        return z
    return (zzz, yyy)