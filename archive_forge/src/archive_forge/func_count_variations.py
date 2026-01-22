from .functions import defun, defun_wrapped
def count_variations(V):
    count = 0
    vold = V[0]
    for n in range(1, len(V)):
        vnew = V[n]
        if vold * vnew < 0:
            count += 1
        vold = vnew
    return count