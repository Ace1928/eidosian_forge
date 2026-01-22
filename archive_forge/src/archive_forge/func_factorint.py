from collections import defaultdict
from functools import reduce
import random
import math
from sympy.core import sympify
from sympy.core.containers import Dict
from sympy.core.evalf import bitcount
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm, Rational, Integer
from sympy.core.power import integer_nthroot, Pow, integer_log
from sympy.core.singleton import S
from sympy.external.gmpy import SYMPY_INTS
from .primetest import isprime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor
def factorint(n, limit=None, use_trial=True, use_rho=True, use_pm1=True, use_ecm=True, verbose=False, visual=None, multiple=False):
    """
    Given a positive integer ``n``, ``factorint(n)`` returns a dict containing
    the prime factors of ``n`` as keys and their respective multiplicities
    as values. For example:

    >>> from sympy.ntheory import factorint
    >>> factorint(2000)    # 2000 = (2**4) * (5**3)
    {2: 4, 5: 3}
    >>> factorint(65537)   # This number is prime
    {65537: 1}

    For input less than 2, factorint behaves as follows:

        - ``factorint(1)`` returns the empty factorization, ``{}``
        - ``factorint(0)`` returns ``{0:1}``
        - ``factorint(-n)`` adds ``-1:1`` to the factors and then factors ``n``

    Partial Factorization:

    If ``limit`` (> 3) is specified, the search is stopped after performing
    trial division up to (and including) the limit (or taking a
    corresponding number of rho/p-1 steps). This is useful if one has
    a large number and only is interested in finding small factors (if
    any). Note that setting a limit does not prevent larger factors
    from being found early; it simply means that the largest factor may
    be composite. Since checking for perfect power is relatively cheap, it is
    done regardless of the limit setting.

    This number, for example, has two small factors and a huge
    semi-prime factor that cannot be reduced easily:

    >>> from sympy.ntheory import isprime
    >>> a = 1407633717262338957430697921446883
    >>> f = factorint(a, limit=10000)
    >>> f == {991: 1, int(202916782076162456022877024859): 1, 7: 1}
    True
    >>> isprime(max(f))
    False

    This number has a small factor and a residual perfect power whose
    base is greater than the limit:

    >>> factorint(3*101**7, limit=5)
    {3: 1, 101: 7}

    List of Factors:

    If ``multiple`` is set to ``True`` then a list containing the
    prime factors including multiplicities is returned.

    >>> factorint(24, multiple=True)
    [2, 2, 2, 3]

    Visual Factorization:

    If ``visual`` is set to ``True``, then it will return a visual
    factorization of the integer.  For example:

    >>> from sympy import pprint
    >>> pprint(factorint(4200, visual=True))
     3  1  2  1
    2 *3 *5 *7

    Note that this is achieved by using the evaluate=False flag in Mul
    and Pow. If you do other manipulations with an expression where
    evaluate=False, it may evaluate.  Therefore, you should use the
    visual option only for visualization, and use the normal dictionary
    returned by visual=False if you want to perform operations on the
    factors.

    You can easily switch between the two forms by sending them back to
    factorint:

    >>> from sympy import Mul
    >>> regular = factorint(1764); regular
    {2: 2, 3: 2, 7: 2}
    >>> pprint(factorint(regular))
     2  2  2
    2 *3 *7

    >>> visual = factorint(1764, visual=True); pprint(visual)
     2  2  2
    2 *3 *7
    >>> print(factorint(visual))
    {2: 2, 3: 2, 7: 2}

    If you want to send a number to be factored in a partially factored form
    you can do so with a dictionary or unevaluated expression:

    >>> factorint(factorint({4: 2, 12: 3})) # twice to toggle to dict form
    {2: 10, 3: 3}
    >>> factorint(Mul(4, 12, evaluate=False))
    {2: 4, 3: 1}

    The table of the output logic is:

        ====== ====== ======= =======
                       Visual
        ------ ----------------------
        Input  True   False   other
        ====== ====== ======= =======
        dict    mul    dict    mul
        n       mul    dict    dict
        mul     mul    dict    dict
        ====== ====== ======= =======

    Notes
    =====

    Algorithm:

    The function switches between multiple algorithms. Trial division
    quickly finds small factors (of the order 1-5 digits), and finds
    all large factors if given enough time. The Pollard rho and p-1
    algorithms are used to find large factors ahead of time; they
    will often find factors of the order of 10 digits within a few
    seconds:

    >>> factors = factorint(12345678910111213141516)
    >>> for base, exp in sorted(factors.items()):
    ...     print('%s %s' % (base, exp))
    ...
    2 2
    2507191691 1
    1231026625769 1

    Any of these methods can optionally be disabled with the following
    boolean parameters:

        - ``use_trial``: Toggle use of trial division
        - ``use_rho``: Toggle use of Pollard's rho method
        - ``use_pm1``: Toggle use of Pollard's p-1 method

    ``factorint`` also periodically checks if the remaining part is
    a prime number or a perfect power, and in those cases stops.

    For unevaluated factorial, it uses Legendre's formula(theorem).


    If ``verbose`` is set to ``True``, detailed progress is printed.

    See Also
    ========

    smoothness, smoothness_p, divisors

    """
    if isinstance(n, Dict):
        n = dict(n)
    if multiple:
        fac = factorint(n, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One / p] * -fac[p] for p in sorted(fac)), [])
        return factorlist
    factordict = {}
    if visual and (not isinstance(n, (Mul, dict))):
        factordict = factorint(n, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False)
    elif isinstance(n, Mul):
        factordict = {int(k): int(v) for k, v in n.as_powers_dict().items()}
    elif isinstance(n, dict):
        factordict = n
    if factordict and isinstance(n, (Mul, dict)):
        for key in list(factordict.keys()):
            if isprime(key):
                continue
            e = factordict.pop(key)
            d = factorint(key, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False)
            for k, v in d.items():
                if k in factordict:
                    factordict[k] += v * e
                else:
                    factordict[k] = v * e
    if visual or (type(n) is dict and visual is not True and (visual is not False)):
        if factordict == {}:
            return S.One
        if -1 in factordict:
            factordict.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False) for i in sorted(factordict.items())])
        return Mul(*args, evaluate=False)
    elif isinstance(n, (dict, Mul)):
        return factordict
    assert use_trial or use_rho or use_pm1 or use_ecm
    from sympy.functions.combinatorial.factorials import factorial
    if isinstance(n, factorial):
        x = as_int(n.args[0])
        if x >= 20:
            factors = {}
            m = 2
            for p in sieve.primerange(2, x + 1):
                if m > 1:
                    m, q = (0, x // p)
                    while q != 0:
                        m += q
                        q //= p
                factors[p] = m
            if factors and verbose:
                for k in sorted(factors):
                    print(factor_msg % (k, factors[k]))
            if verbose:
                print(complete_msg)
            return factors
        else:
            n = n.func(x)
    n = as_int(n)
    if limit:
        limit = int(limit)
        use_ecm = False
    if n < 0:
        factors = factorint(-n, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False)
        factors[-1] = 1
        return factors
    if limit and limit < 2:
        if n == 1:
            return {}
        return {n: 1}
    elif n < 10:
        return [{0: 1}, {}, {2: 1}, {3: 1}, {2: 2}, {5: 1}, {2: 1, 3: 1}, {7: 1}, {2: 3}, {3: 2}][n]
    factors = {}
    if verbose:
        sn = str(n)
        if len(sn) > 50:
            print('Factoring %s' % sn[:5] + '..(%i other digits)..' % (len(sn) - 10) + sn[-5:])
        else:
            print('Factoring', n)
    if use_trial:
        small = 2 ** 15
        fail_max = 600
        small = min(small, limit or small)
        if verbose:
            print(trial_int_msg % (2, small, fail_max))
        n, next_p = _factorint_small(factors, n, small, fail_max)
    else:
        next_p = 2
    if factors and verbose:
        for k in sorted(factors):
            print(factor_msg % (k, factors[k]))
    if next_p == 0:
        if n > 1:
            factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return factors
    try:
        if limit and next_p > limit:
            if verbose:
                print('Exceeded limit:', limit)
            _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose)
            if n > 1:
                factors[int(n)] = 1
            return factors
        else:
            sqrt_n = integer_nthroot(n, 2)[0]
            a = sqrt_n + 1
            a2 = a ** 2
            b2 = a2 - n
            for i in range(3):
                b, fermat = integer_nthroot(b2, 2)
                if fermat:
                    break
                b2 += 2 * a + 1
                a += 1
            if fermat:
                if verbose:
                    print(fermat_msg)
                if limit:
                    limit -= 1
                for r in [a - b, a + b]:
                    facs = factorint(r, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose)
                    for k, v in facs.items():
                        factors[k] = factors.get(k, 0) + v
                raise StopIteration
            _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose)
    except StopIteration:
        if verbose:
            print(complete_msg)
        return factors
    low, high = (next_p, 2 * next_p)
    limit = limit or sqrt_n
    limit += 1
    iteration = 0
    while 1:
        try:
            high_ = high
            if limit < high_:
                high_ = limit
            if use_trial:
                if verbose:
                    print(trial_msg % (low, high_))
                ps = sieve.primerange(low, high_)
                n, found_trial = _trial(factors, n, ps, verbose)
                if found_trial:
                    _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose)
            else:
                found_trial = False
            if high > limit:
                if verbose:
                    print('Exceeded limit:', limit)
                if n > 1:
                    factors[int(n)] = 1
                raise StopIteration
            if not found_trial:
                if use_pm1 or use_rho:
                    high_root = max(int(math.log(high_ ** 0.7)), low, 3)
                    if use_pm1:
                        if verbose:
                            print(pm1_msg % (high_root, high_))
                        c = pollard_pm1(n, B=high_root, seed=high_)
                        if c:
                            ps = factorint(c, limit=limit - 1, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, use_ecm=use_ecm, verbose=verbose)
                            n, _ = _trial(factors, n, ps, verbose=False)
                            _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose)
                    if use_rho:
                        max_steps = high_root
                        if verbose:
                            print(rho_msg % (1, max_steps, high_))
                        c = pollard_rho(n, retries=1, max_steps=max_steps, seed=high_)
                        if c:
                            ps = factorint(c, limit=limit - 1, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, use_ecm=use_ecm, verbose=verbose)
                            n, _ = _trial(factors, n, ps, verbose=False)
                            _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose)
        except StopIteration:
            if verbose:
                print(complete_msg)
            return factors
        iteration += 1
        if use_ecm and iteration >= 3 and (len(str(n)) >= 25):
            break
        low, high = (high, high * 2)
    B1 = 10000
    B2 = 100 * B1
    num_curves = 50
    while 1:
        if verbose:
            print(ecm_msg % (B1, B2, num_curves))
        while 1:
            try:
                factor = _ecm_one_factor(n, B1, B2, num_curves)
                ps = factorint(factor, limit=limit - 1, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, use_ecm=use_ecm, verbose=verbose)
                n, _ = _trial(factors, n, ps, verbose=False)
                _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose)
            except ValueError:
                break
            except StopIteration:
                if verbose:
                    print(complete_msg)
                return factors
        B1 *= 5
        B2 = 100 * B1
        num_curves *= 4