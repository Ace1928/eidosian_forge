import atexit
import pyglet
def get_audio_device_manager():
    global _audio_device_manager
    if _audio_device_manager:
        return _audio_device_manager
    _audio_device_manager = None
    if pyglet.compat_platform == 'win32':
        from pyglet.media.devices.win32 import Win32AudioDeviceManager
        _audio_device_manager = Win32AudioDeviceManager()
    return _audio_device_manager