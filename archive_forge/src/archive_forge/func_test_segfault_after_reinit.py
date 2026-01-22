from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_segfault_after_reinit(self):
    """Reinitialization of font module should not cause
        segmentation fault"""
    import gc
    font = pygame_font.Font(None, 20)
    pygame_font.quit()
    pygame_font.init()
    del font
    gc.collect()