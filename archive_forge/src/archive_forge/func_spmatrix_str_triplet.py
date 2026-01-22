def spmatrix_str_triplet(X):
    from cvxopt.printing import options
    iformat, dformat = (options['iformat'], options['dformat'])
    sgn = ['-', '+']
    if X.typecode == 'i':
        fmt = iformat
    else:
        fmt = dformat
    s = ''
    if len(X) > 0:
        if X.typecode == 'z':
            twidth = max([len(fmt % Xk.real + sgn[Xk.imag > 0] + 'j' + (fmt % abs(Xk.imag)).lstrip()) for Xk in X.V])
        else:
            twidth = max([len(fmt % Xk) for Xk in X.V])
        imax = max([len(str(i)) for i in X.I])
        jmax = max([len(str(j)) for j in X.J])
    else:
        twidth = 0
    for k in range(len(X)):
        s += '('
        s += format(X.I[k], '>%i' % imax) + ',' + format(X.J[k], '>%i' % jmax)
        s += ') '
        if X.typecode == 'z':
            s += format(fmt % X.V[k].real + sgn[X.V[k].imag > 0] + 'j' + (fmt % abs(X.V[k].imag)).lstrip(), '>%i' % twidth)
        else:
            s += format(fmt % X.V[k], '>%i' % twidth)
        s += '\n'
    return s