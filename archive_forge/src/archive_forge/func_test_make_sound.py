import unittest
from numpy import int8, int16, uint8, uint16, float32, array, alltrue
import pygame
import pygame.sndarray
def test_make_sound(self):

    def check_sound(size, channels, test_data):
        try:
            pygame.mixer.init(22050, size, channels, allowedchanges=0)
        except pygame.error:
            return
        try:
            __, sz, __ = pygame.mixer.get_init()
            if sz == size:
                srcarr = array(test_data, self.array_dtypes[size])
                snd = pygame.sndarray.make_sound(srcarr)
                arr = pygame.sndarray.samples(snd)
                self.assertTrue(alltrue(arr == srcarr), 'size: %i\n%s\n%s' % (size, arr, test_data))
        finally:
            pygame.mixer.quit()
    check_sound(8, 1, [0, 15, 240, 255])
    check_sound(8, 2, [[0, 128], [45, 65], [100, 161], [255, 64]])
    check_sound(16, 1, [0, 255, 65280, 65535])
    check_sound(16, 2, [[0, 65535], [65535, 0], [255, 65280], [3855, 61680]])
    check_sound(-8, 1, [0, -128, 127, 100])
    check_sound(-8, 2, [[0, -128], [-100, 100], [37, -80], [255, 0]])
    check_sound(-16, 1, [0, 32767, -32767, -1])
    check_sound(-16, 2, [[0, -32767], [-32767, 0], [32767, 0], [0, 32767]])
    check_sound(32, 2, [[0.0, -1.0], [-1.0, 0], [1.0, 0], [0, 1.0]])