import os
import pygame
import sys
import tempfile
import time
def trunk_relative_path(relative):
    return os.path.normpath(os.path.join(trunk_dir, relative))