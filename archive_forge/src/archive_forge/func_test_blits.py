import unittest
import pygame
from pygame.locals import *
def test_blits(self):
    NUM_SURFS = 255
    PRINT_TIMING = 0
    dst = pygame.Surface((NUM_SURFS * 10, 10), SRCALPHA, 32)
    dst.fill((230, 230, 230))
    blit_list = self.make_blit_list(NUM_SURFS)

    def blits(blit_list):
        for surface, dest in blit_list:
            dst.blit(surface, dest)
    from time import time
    t0 = time()
    results = blits(blit_list)
    t1 = time()
    if PRINT_TIMING:
        print(f'python blits: {t1 - t0}')
    dst.fill((230, 230, 230))
    t0 = time()
    results = dst.blits(blit_list)
    t1 = time()
    if PRINT_TIMING:
        print(f'Surface.blits :{t1 - t0}')
    for i in range(NUM_SURFS):
        color = (i * 1, i * 1, i * 1)
        self.assertEqual(dst.get_at((i * 10, 0)), color)
        self.assertEqual(dst.get_at((i * 10 + 5, 5)), color)
    self.assertEqual(len(results), NUM_SURFS)
    t0 = time()
    results = dst.blits(blit_list, doreturn=0)
    t1 = time()
    if PRINT_TIMING:
        print(f'Surface.blits doreturn=0: {t1 - t0}')
    self.assertEqual(results, None)
    t0 = time()
    results = dst.blits(((surf, dest) for surf, dest in blit_list))
    t1 = time()
    if PRINT_TIMING:
        print(f'Surface.blits generator: {t1 - t0}')