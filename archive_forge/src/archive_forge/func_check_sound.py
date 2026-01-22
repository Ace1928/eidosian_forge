import unittest
from numpy import int8, int16, uint8, uint16, float32, array, alltrue
import pygame
import pygame.sndarray
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