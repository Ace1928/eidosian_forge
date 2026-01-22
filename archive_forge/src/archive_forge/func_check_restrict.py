import textwrap
def check_restrict(cmd):
    """Return the restrict identifier (may be empty)."""
    cmd._check_compiler()
    body = textwrap.dedent('\n        static int static_func (char * %(restrict)s a)\n        {\n            return 0;\n        }\n        ')
    for kw in ['restrict', '__restrict__', '__restrict']:
        st = cmd.try_compile(body % {'restrict': kw}, None, None)
        if st:
            return kw
    return ''