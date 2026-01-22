def shellcode(executable, use_defaults=True, shell='bash', complete_arguments=None):
    """
    Provide the shell code required to register a python executable for use with the argcomplete module.

    :param str executable: Executable to be completed (when invoked exactly with this name
    :param bool use_defaults: Whether to fallback to readline's default completion when no matches are generated.
    :param str shell: Name of the shell to output code for (bash or tcsh)
    :param complete_arguments: Arguments to call complete with
    :type complete_arguments: list(str) or None
    """
    if complete_arguments is None:
        complete_options = '-o nospace -o default' if use_defaults else '-o nospace'
    else:
        complete_options = ' '.join(complete_arguments)
    if shell == 'bash':
        code = bashcode
    else:
        code = tcshcode
    return code % dict(complete_opts=complete_options, executable=executable)