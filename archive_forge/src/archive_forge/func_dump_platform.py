def dump_platform():
    """Dump OS specific """
    import platform
    print('platform: ', platform.platform())
    print('release:  ', platform.release())
    print('machine:  ', platform.machine())