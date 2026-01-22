import os
def gnu_getopt(args, shortopts, longopts=[]):
    """getopt(args, options[, long_options]) -> opts, args

    This function works like getopt(), except that GNU style scanning
    mode is used by default. This means that option and non-option
    arguments may be intermixed. The getopt() function stops
    processing options as soon as a non-option argument is
    encountered.

    If the first character of the option string is `+', or if the
    environment variable POSIXLY_CORRECT is set, then option
    processing stops as soon as a non-option argument is encountered.

    """
    opts = []
    prog_args = []
    if isinstance(longopts, str):
        longopts = [longopts]
    else:
        longopts = list(longopts)
    if shortopts.startswith('+'):
        shortopts = shortopts[1:]
        all_options_first = True
    elif os.environ.get('POSIXLY_CORRECT'):
        all_options_first = True
    else:
        all_options_first = False
    while args:
        if args[0] == '--':
            prog_args += args[1:]
            break
        if args[0][:2] == '--':
            opts, args = do_longs(opts, args[0][2:], longopts, args[1:])
        elif args[0][:1] == '-' and args[0] != '-':
            opts, args = do_shorts(opts, args[0][1:], shortopts, args[1:])
        elif all_options_first:
            prog_args += args
            break
        else:
            prog_args.append(args[0])
            args = args[1:]
    return (opts, prog_args)