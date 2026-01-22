def dump_pyglet():
    """Dump pyglet version and options."""
    import pyglet
    print('pyglet.version:', pyglet.version)
    print('pyglet.compat_platform:', pyglet.compat_platform)
    print('pyglet.__file__:', pyglet.__file__)
    for key, value in pyglet.options.items():
        print(f"pyglet.options['{key}'] = {value!r}")