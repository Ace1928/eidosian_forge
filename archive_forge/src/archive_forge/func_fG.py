import sys
def fG(x, y, trans='N', alpha=1.0, beta=0.0):
    misc.sgemv(G, x, y, dims, trans=trans, alpha=alpha, beta=beta)