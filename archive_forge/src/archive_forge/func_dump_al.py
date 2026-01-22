def dump_al():
    """Dump OpenAL info."""
    try:
        from pyglet.media.drivers import openal
    except:
        print('OpenAL not available.')
        return
    print('Library:', openal.lib_openal._lib)
    driver = openal.create_audio_driver()
    print('Version: {}.{}'.format(*driver.get_version()))
    print('Extensions:')
    for extension in driver.get_extensions():
        print('  ', extension)