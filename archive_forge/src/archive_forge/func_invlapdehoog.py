def invlapdehoog(ctx, *args, **kwargs):
    kwargs['method'] = 'dehoog'
    return ctx.invertlaplace(*args, **kwargs)