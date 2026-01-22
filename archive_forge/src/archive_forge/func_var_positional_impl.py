def var_positional_impl(a, *star_args_token, kw=None, kw1=12):

    def impl(a, b, f, kw=None, kw1=12):
        if a > 10:
            return 1
        else:
            return -1
    return impl