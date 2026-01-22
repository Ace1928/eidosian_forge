import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
@unittest.skipIf(sys.platform.startswith('win'), 'See github issue 892.')
@unittest.skipIf(IS_PYPY, 'random errors here with pypy')
def test_sound_args(self):

    def get_bytes(snd):
        return snd.get_raw()
    mixer.init()
    sample = b'\x00\xff' * 24
    wave_path = example_path(os.path.join('data', 'house_lo.wav'))
    uwave_path = str(wave_path)
    bwave_path = uwave_path.encode(sys.getfilesystemencoding())
    snd = mixer.Sound(file=wave_path)
    self.assertTrue(snd.get_length() > 0.5)
    snd_bytes = get_bytes(snd)
    self.assertTrue(len(snd_bytes) > 1000)
    self.assertEqual(get_bytes(mixer.Sound(wave_path)), snd_bytes)
    self.assertEqual(get_bytes(mixer.Sound(file=uwave_path)), snd_bytes)
    self.assertEqual(get_bytes(mixer.Sound(uwave_path)), snd_bytes)
    arg_emsg = 'Sound takes either 1 positional or 1 keyword argument'
    with self.assertRaises(TypeError) as cm:
        mixer.Sound()
    self.assertEqual(str(cm.exception), arg_emsg)
    with self.assertRaises(TypeError) as cm:
        mixer.Sound(wave_path, buffer=sample)
    self.assertEqual(str(cm.exception), arg_emsg)
    with self.assertRaises(TypeError) as cm:
        mixer.Sound(sample, file=wave_path)
    self.assertEqual(str(cm.exception), arg_emsg)
    with self.assertRaises(TypeError) as cm:
        mixer.Sound(buffer=sample, file=wave_path)
    self.assertEqual(str(cm.exception), arg_emsg)
    with self.assertRaises(TypeError) as cm:
        mixer.Sound(foobar=sample)
    self.assertEqual(str(cm.exception), "Unrecognized keyword argument 'foobar'")
    snd = mixer.Sound(wave_path, **{})
    self.assertEqual(get_bytes(snd), snd_bytes)
    snd = mixer.Sound(*[], **{'file': wave_path})
    with self.assertRaises(TypeError) as cm:
        mixer.Sound([])
    self.assertEqual(str(cm.exception), 'Unrecognized argument (type list)')
    with self.assertRaises(TypeError) as cm:
        snd = mixer.Sound(buffer=[])
    emsg = 'Expected object with buffer interface: got a list'
    self.assertEqual(str(cm.exception), emsg)
    ufake_path = '12345678'
    self.assertRaises(IOError, mixer.Sound, ufake_path)
    self.assertRaises(IOError, mixer.Sound, '12345678')
    with self.assertRaises(TypeError) as cm:
        mixer.Sound(buffer='something')
    emsg = 'Unicode object not allowed as buffer object'
    self.assertEqual(str(cm.exception), emsg)
    self.assertEqual(get_bytes(mixer.Sound(buffer=sample)), sample)
    if type(sample) != str:
        somebytes = get_bytes(mixer.Sound(sample))
        self.assertEqual(somebytes, sample)
    self.assertEqual(get_bytes(mixer.Sound(file=bwave_path)), snd_bytes)
    self.assertEqual(get_bytes(mixer.Sound(bwave_path)), snd_bytes)
    snd = mixer.Sound(wave_path)
    with self.assertRaises(TypeError) as cm:
        mixer.Sound(wave_path, array=snd)
    self.assertEqual(str(cm.exception), arg_emsg)
    with self.assertRaises(TypeError) as cm:
        mixer.Sound(buffer=sample, array=snd)
    self.assertEqual(str(cm.exception), arg_emsg)
    snd2 = mixer.Sound(array=snd)
    self.assertEqual(snd.get_raw(), snd2.get_raw())