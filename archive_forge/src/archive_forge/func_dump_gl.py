def dump_gl(context=None):
    """Dump GL info."""
    if context is not None:
        info = context.get_info()
    else:
        from pyglet.gl import gl_info as info
    print('gl_info.get_version():', info.get_version())
    print('gl_info.get_vendor():', info.get_vendor())
    print('gl_info.get_renderer():', info.get_renderer())