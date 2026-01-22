from typing import List
import pygame as pg
import os, sys
def change_music_position(amount):
    """
    Changes current playback position by amount seconds.
    This only works with OGG and MP3 files.
    music.get_pos() returns how many milliseconds the song has played, not
    the current position in the file. We must track the starting position
    ourselves. music.set_pos() will set the position in seconds.
    """
    global starting_pos
    if starting_pos >= 0:
        played_for = pg.mixer.music.get_pos() / 1000.0
        old_pos = starting_pos + played_for
        starting_pos = old_pos + amount
        pg.mixer.music.play(start=starting_pos)
        print(f'jumped from {old_pos} to {starting_pos}')