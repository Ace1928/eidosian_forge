def default_config_filename(installer):
    """
    This function can return a default filename or directory for the
    configuration file, if none was explicitly given.

    Return None to mean no preference.  The first non-None returning
    value will be used.

    Pay attention to ``installer.expect_config_directory`` here,
    and to ``installer.default_config_filename``.
    """
    return installer.default_config_filename