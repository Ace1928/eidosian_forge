from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def hypercomb(ctx, function, params=[], discard_known_zeros=True, **kwargs):
    orig = ctx.prec
    sumvalue = ctx.zero
    dist = ctx.nint_distance
    ninf = ctx.ninf
    orig_params = params[:]
    verbose = kwargs.get('verbose', False)
    maxprec = kwargs.get('maxprec', ctx._default_hyper_maxprec(orig))
    kwargs['maxprec'] = maxprec
    zeroprec = kwargs.get('zeroprec')
    infprec = kwargs.get('infprec')
    perturbed_reference_value = None
    hextra = 0
    try:
        while 1:
            ctx.prec += 10
            if ctx.prec > maxprec:
                raise ValueError(_hypercomb_msg % (orig, ctx.prec))
            orig2 = ctx.prec
            params = orig_params[:]
            terms = function(*params)
            if verbose:
                print()
                print('ENTERING hypercomb main loop')
                print('prec =', ctx.prec)
                print('hextra', hextra)
            perturb, recompute, extraprec, discard = _check_need_perturb(ctx, terms, orig, discard_known_zeros)
            ctx.prec += extraprec
            if perturb:
                if 'hmag' in kwargs:
                    hmag = kwargs['hmag']
                elif ctx._fixed_precision:
                    hmag = int(ctx.prec * 0.3)
                else:
                    hmag = orig + 10 + hextra
                h = ctx.ldexp(ctx.one, -hmag)
                ctx.prec = orig2 + 10 + hmag + 10
                for k in range(len(params)):
                    params[k] += h
                    h += h / (k + 1)
            if recompute:
                terms = function(*params)
            if discard_known_zeros:
                terms = [term for i, term in enumerate(terms) if i not in discard]
            if not terms:
                return ctx.zero
            evaluated_terms = []
            for term_index, term_data in enumerate(terms):
                w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term_data
                if verbose:
                    print()
                    print('  Evaluating term %i/%i : %iF%i' % (term_index + 1, len(terms), len(a_s), len(b_s)))
                    print('    powers', ctx.nstr(w_s), ctx.nstr(c_s))
                    print('    gamma', ctx.nstr(alpha_s), ctx.nstr(beta_s))
                    print('    hyper', ctx.nstr(a_s), ctx.nstr(b_s))
                    print('    z', ctx.nstr(z))
                v = ctx.fprod([ctx.hyper(a_s, b_s, z, **kwargs)] + [ctx.gamma(a) for a in alpha_s] + [ctx.rgamma(b) for b in beta_s] + [ctx.power(w, c) for w, c in zip(w_s, c_s)])
                if verbose:
                    print('    Value:', v)
                evaluated_terms.append(v)
            if len(terms) == 1 and (not perturb):
                sumvalue = evaluated_terms[0]
                break
            if ctx._fixed_precision:
                sumvalue = ctx.fsum(evaluated_terms)
                break
            sumvalue = ctx.fsum(evaluated_terms)
            term_magnitudes = [ctx.mag(x) for x in evaluated_terms]
            max_magnitude = max(term_magnitudes)
            sum_magnitude = ctx.mag(sumvalue)
            cancellation = max_magnitude - sum_magnitude
            if verbose:
                print()
                print('  Cancellation:', cancellation, 'bits')
                print('  Increased precision:', ctx.prec - orig, 'bits')
            precision_ok = cancellation < ctx.prec - orig
            if zeroprec is None:
                zero_ok = False
            else:
                zero_ok = max_magnitude - ctx.prec < -zeroprec
            if infprec is None:
                inf_ok = False
            else:
                inf_ok = max_magnitude > infprec
            if precision_ok and (not perturb) or ctx.isnan(cancellation):
                break
            elif precision_ok:
                if perturbed_reference_value is None:
                    hextra += 20
                    perturbed_reference_value = sumvalue
                    continue
                elif ctx.mag(sumvalue - perturbed_reference_value) <= ctx.mag(sumvalue) - orig:
                    break
                elif zero_ok:
                    sumvalue = ctx.zero
                    break
                elif inf_ok:
                    sumvalue = ctx.inf
                    break
                elif 'hmag' in kwargs:
                    break
                else:
                    hextra *= 2
                    perturbed_reference_value = sumvalue
            else:
                increment = min(max(cancellation, orig // 2), max(extraprec, orig))
                ctx.prec += increment
                if verbose:
                    print('  Must start over with increased precision')
                continue
    finally:
        ctx.prec = orig
    return +sumvalue