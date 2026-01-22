from typing import List
import pygame as pg
import pygame._sdl2.controller
def showtext(win, pos, text, color, bgcolor):
    textimg = font.render(text, 1, color, bgcolor)
    win.blit(textimg, pos)
    return (pos[0] + textimg.get_width() + 5, pos[1])