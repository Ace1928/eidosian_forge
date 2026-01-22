import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_correct_gamma__verified_against_python_implementation(self):
    """|tags:slow|"""
    gammas = [i / 10.0 for i in range(1, 31)]
    gammas_len = len(gammas)
    for i, c in enumerate(rgba_combos_Color_generator()):
        gamma = gammas[i % gammas_len]
        corrected = pygame.Color(*[gamma_correct(x, gamma) for x in tuple(c)])
        lib_corrected = c.correct_gamma(gamma)
        self.assertTrue(corrected.r == lib_corrected.r)
        self.assertTrue(corrected.g == lib_corrected.g)
        self.assertTrue(corrected.b == lib_corrected.b)
        self.assertTrue(corrected.a == lib_corrected.a)