import textwrap
def check_gcc_variable_attribute(cmd, attribute):
    """Return True if the given variable attribute is supported."""
    cmd._check_compiler()
    body = textwrap.dedent('\n        #pragma GCC diagnostic error "-Wattributes"\n        #pragma clang diagnostic error "-Wattributes"\n\n        int %s foo;\n\n        int\n        main()\n        {\n            return 0;\n        }\n        ') % (attribute,)
    return cmd.try_compile(body, None, None) != 0