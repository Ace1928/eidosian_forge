from typing import List
import pygame as pg
import pygame._sdl2.controller
def draw_usage_in_history(history, text):
    lines = text.split('\n')
    for line in lines:
        if line == '' or '===' in line:
            continue
        img = font.render(line, 1, (50, 200, 50), (0, 0, 0))
        history.append(img)