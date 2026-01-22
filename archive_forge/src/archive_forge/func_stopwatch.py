import os
import pygame as pg
def stopwatch(message=None):
    """simple routine to time python code"""
    global timer
    if not message:
        timer = pg.time.get_ticks()
        return
    now = pg.time.get_ticks()
    runtime = (now - timer) / 1000.0 + 0.001
    print(f'{message} {runtime} seconds\t{1.0 / runtime:.2f}fps')
    timer = now