from .functions import defun, defun_wrapped
def separate_zeros_in_block(ctx, zero_number_block, T, V, limitloop=None, fp_tolerance=None):
    """Separate the zeros contained in the block T, limitloop
    determines how long one must search"""
    if limitloop is None:
        limitloop = ctx.inf
    loopnumber = 0
    variations = count_variations(V)
    while variations < zero_number_block and loopnumber < limitloop:
        a = T[0]
        v = V[0]
        newT = [a]
        newV = [v]
        variations = 0
        for n in range(1, len(T)):
            b2 = T[n]
            u = V[n]
            if u * v > 0:
                alpha = ctx.sqrt(u / v)
                b = (alpha * a + b2) / (alpha + 1)
            else:
                b = (a + b2) / 2
            if fp_tolerance < 10:
                w = ctx._fp.siegelz(b)
                if abs(w) < fp_tolerance:
                    w = ctx.siegelz(b)
            else:
                w = ctx.siegelz(b)
            if v * w < 0:
                variations += 1
            newT.append(b)
            newV.append(w)
            u = V[n]
            if u * w < 0:
                variations += 1
            newT.append(b2)
            newV.append(u)
            a = b2
            v = u
        T = newT
        V = newV
        loopnumber += 1
        if limitloop > ITERATION_LIMIT and loopnumber > 2 and (variations + 2 == zero_number_block):
            dtMax = 0
            dtSec = 0
            kMax = 0
            for k1 in range(1, len(T)):
                dt = T[k1] - T[k1 - 1]
                if dt > dtMax:
                    kMax = k1
                    dtSec = dtMax
                    dtMax = dt
                elif dt < dtMax and dt > dtSec:
                    dtSec = dt
            if dtMax > 3 * dtSec:
                f = lambda x: ctx.rs_z(x, derivative=1)
                t0 = T[kMax - 1]
                t1 = T[kMax]
                t = ctx.findroot(f, (t0, t1), solver='illinois', verify=False, verbose=False)
                v = ctx.siegelz(t)
                if t0 < t and t < t1 and (v * V[kMax] < 0):
                    T.insert(kMax, t)
                    V.insert(kMax, v)
        variations = count_variations(V)
    if variations == zero_number_block:
        separated = True
    else:
        separated = False
    return (T, V, separated)