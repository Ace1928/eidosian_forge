import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
PATCH: pygame.scrap on X11, fix copying into PRIMARY selection

        Copying into theX11 PRIMARY selection (mouse copy/paste) would not
        work due to a confusion between content type and clipboard type.

        