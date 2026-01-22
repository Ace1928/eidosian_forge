import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
@unittest.skipIf(os.environ.get('SDL_AUDIODRIVER') == 'disk', 'disk audio driver "playback" writing to disk is slow')
def test_play__start_time(self):
    pygame.display.init()
    filename = example_path(os.path.join('data', 'house_lo.ogg'))
    pygame.mixer.music.load(filename)
    start_time_in_seconds = 6.0
    music_finished = False
    clock = pygame.time.Clock()
    start_time_in_ms = clock.tick()
    pygame.mixer.music.play(0, start=start_time_in_seconds)
    running = True
    while running:
        pygame.event.pump()
        if not (pygame.mixer.music.get_busy() or music_finished):
            music_finished = True
            time_to_finish = (clock.tick() - start_time_in_ms) // 1000
            self.assertEqual(time_to_finish, 1)
            running = False