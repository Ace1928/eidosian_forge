from typing import List
import pygame as pg
import os, sys
def play_next():
    """
    This function will play the next song in music_file_list
    It uses pop(0) to get the next song and then appends it to the end of the list
    The song will fade in during the first 4 seconds
    """
    global starting_pos
    if len(music_file_list) > 1:
        nxt = music_file_list.pop(0)
        try:
            pg.mixer.music.load(nxt)
        except pg.error as e:
            print(e)
            print(f'{nxt} removed from file list')
        music_file_list.append(nxt)
        print('starting next song: ', nxt)
    else:
        nxt = music_file_list[0]
    pg.mixer.music.play(fade_ms=4000)
    pg.mixer.music.set_volume(volume)
    pg.mixer.music.set_endevent(MUSIC_DONE)
    if nxt.rpartition('.')[2].lower() in music_can_seek:
        starting_pos = 0
    else:
        starting_pos = -1