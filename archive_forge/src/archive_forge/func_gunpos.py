import os
import random
from typing import List
import pygame as pg
def gunpos(self):
    pos = self.facing * self.gun_offset + self.rect.centerx
    return (pos, self.rect.top)