def dump_glx():
    """Dump GLX info."""
    try:
        from pyglet.gl import glx_info
    except:
        print('GLX not available.')
        return
    import pyglet
    window = pyglet.window.Window(visible=False)
    print('context.is_direct():', window.context.is_direct())
    window.close()
    if not glx_info.have_version(1, 1):
        print('Version: < 1.1')
    else:
        print('glx_info.get_server_vendor():', glx_info.get_server_vendor())
        print('glx_info.get_server_version():', glx_info.get_server_version())
        print('glx_info.get_server_extensions():')
        for name in glx_info.get_server_extensions():
            print('  ', name)
        print('glx_info.get_client_vendor():', glx_info.get_client_vendor())
        print('glx_info.get_client_version():', glx_info.get_client_version())
        print('glx_info.get_client_extensions():')
        for name in glx_info.get_client_extensions():
            print('  ', name)
        print('glx_info.get_extensions():')
        for name in glx_info.get_extensions():
            print('  ', name)