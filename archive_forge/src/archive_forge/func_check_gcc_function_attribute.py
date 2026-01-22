import textwrap
def check_gcc_function_attribute(cmd, attribute, name):
    """Return True if the given function attribute is supported."""
    cmd._check_compiler()
    body = textwrap.dedent('\n        #pragma GCC diagnostic error "-Wattributes"\n        #pragma clang diagnostic error "-Wattributes"\n\n        int %s %s(void* unused)\n        {\n            return 0;\n        }\n\n        int\n        main()\n        {\n            return 0;\n        }\n        ') % (attribute, name)
    return cmd.try_compile(body, None, None) != 0