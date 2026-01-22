import sys
def shellcomplete_on_options(options, outfile=None):
    for opt in options:
        short_name = opt.short_name()
        if short_name:
            outfile.write('"(--%s -%s)"{--%s,-%s}\n' % (opt.name, short_name, opt.name, short_name))
        else:
            outfile.write('--%s\n' % opt.name)