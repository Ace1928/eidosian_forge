def dump_media():
    """Dump pyglet.media info."""
    import pyglet.media
    print('audio driver:', pyglet.media.get_audio_driver())