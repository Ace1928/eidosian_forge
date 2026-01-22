from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def modgcd_bivariate(f, g):
    """
    Computes the GCD of two polynomials in `\\mathbb{Z}[x, y]` using a
    modular algorithm.

    The algorithm computes the GCD of two bivariate integer polynomials
    `f` and `g` by calculating the GCD in `\\mathbb{Z}_p[x, y]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem. To compute the bivariate GCD over
    `\\mathbb{Z}_p`, the polynomials `f \\; \\mathrm{mod} \\, p` and
    `g \\; \\mathrm{mod} \\, p` are evaluated at `y = a` for certain
    `a \\in \\mathbb{Z}_p` and then their univariate GCD in `\\mathbb{Z}_p[x]`
    is computed. Interpolating those yields the bivariate GCD in
    `\\mathbb{Z}_p[x, y]`. To verify the result in `\\mathbb{Z}[x, y]`, trial
    division is done, but only for candidates which are very likely the
    desired GCD.

    Parameters
    ==========

    f : PolyElement
        bivariate integer polynomial
    g : PolyElement
        bivariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_bivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2 - y**2
    >>> g = x**2 + 2*x*y + y**2

    >>> h, cff, cfg = modgcd_bivariate(f, g)
    >>> h, cff, cfg
    (x + y, x - y, x + y)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = x**2*y - x**2 - 4*y + 4
    >>> g = x + 2

    >>> h, cff, cfg = modgcd_bivariate(f, g)
    >>> h, cff, cfg
    (x + 2, x*y - x - 2*y + 2, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ
    result = _trivial_gcd(f, g)
    if result is not None:
        return result
    ring = f.ring
    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)
    xbound, ycontbound = _degree_bound_bivariate(f, g)
    if xbound == ycontbound == 0:
        return (ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch))
    fswap = _swap(f, 1)
    gswap = _swap(g, 1)
    degyf = fswap.degree()
    degyg = gswap.degree()
    ybound, xcontbound = _degree_bound_bivariate(fswap, gswap)
    if ybound == xcontbound == 0:
        return (ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch))
    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(fswap.LC, gswap.LC)
    badprimes = gamma1 * gamma2
    m = 1
    p = 1
    while True:
        p = nextprime(p)
        while badprimes % p == 0:
            p = nextprime(p)
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        contfp, fp = _primitive(fp, p)
        contgp, gp = _primitive(gp, p)
        conthp = _gf_gcd(contfp, contgp, p)
        degconthp = conthp.degree()
        if degconthp > ycontbound:
            continue
        elif degconthp < ycontbound:
            m = 1
            ycontbound = degconthp
            continue
        delta = _gf_gcd(_LC(fp), _LC(gp), p)
        degcontfp = contfp.degree()
        degcontgp = contgp.degree()
        degdelta = delta.degree()
        N = min(degyf - degcontfp, degyg - degcontgp, ybound - ycontbound + degdelta) + 1
        if p < N:
            continue
        n = 0
        evalpoints = []
        hpeval = []
        unlucky = False
        for a in range(p):
            deltaa = delta.evaluate(0, a)
            if not deltaa % p:
                continue
            fpa = fp.evaluate(1, a).trunc_ground(p)
            gpa = gp.evaluate(1, a).trunc_ground(p)
            hpa = _gf_gcd(fpa, gpa, p)
            deghpa = hpa.degree()
            if deghpa > xbound:
                continue
            elif deghpa < xbound:
                m = 1
                xbound = deghpa
                unlucky = True
                break
            hpa = hpa.mul_ground(deltaa).trunc_ground(p)
            evalpoints.append(a)
            hpeval.append(hpa)
            n += 1
            if n == N:
                break
        if unlucky:
            continue
        if n < N:
            continue
        hp = _interpolate_multivariate(evalpoints, hpeval, ring, 1, p)
        hp = _primitive(hp, p)[1]
        hp = hp * conthp.set_ring(ring)
        degyhp = hp.degree(1)
        if degyhp > ybound:
            continue
        if degyhp < ybound:
            m = 1
            ybound = degyhp
            continue
        hp = hp.mul_ground(gamma1).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue
        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        m *= p
        if not hm == hlastm:
            hlastm = hm
            continue
        h = hm.quo_ground(hm.content())
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and (not grem):
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return (h, cff, cfg)