def dump_ffmpeg():
    """Dump FFmpeg info."""
    import pyglet
    pyglet.options['search_local_libs'] = True
    import pyglet.media
    if pyglet.media.have_ffmpeg():
        from pyglet.media.codecs.ffmpeg import get_version
        print('FFmpeg version:', get_version())
    else:
        print('FFmpeg not available.')