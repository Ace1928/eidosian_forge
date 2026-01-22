import os
import pygame as pg
from pygame import surfarray
def surfdemo_show(array_img, name):
    """displays a surface, waits for user to continue"""
    screen = pg.display.set_mode(array_img.shape[:2], 0, 32)
    surfarray.blit_array(screen, array_img)
    pg.display.flip()
    pg.display.set_caption(name)
    while True:
        e = pg.event.wait()
        if e.type == pg.MOUSEBUTTONUP and e.button == pg.BUTTON_LEFT:
            break
        elif e.type == pg.KEYDOWN and e.key == pg.K_s:
            pg.image.save(screen, name + '.png')
        elif e.type == pg.QUIT:
            pg.quit()
            raise SystemExit()