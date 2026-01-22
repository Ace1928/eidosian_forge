from typing import List
import pygame as pg
import pygame._sdl2.controller
def drawstatus(win):
    global virtual_x, virtual_y
    bgcolor = (50, 50, 50)
    win.fill(bgcolor, (0, 0, 640, 120))
    win.blit(font.render('Status Area', 1, (155, 155, 155), bgcolor), (2, 2))
    pos = showtext(win, (10, 30), 'Mouse Focus', (255, 255, 255), bgcolor)
    win.blit(img_on_off[pg.mouse.get_focused()], pos)
    pos = showtext(win, (pos[0] + 50, pos[1]), 'Mouse visible', (255, 255, 255), bgcolor)
    win.blit(img_on_off[pg.mouse.get_visible()], pos)
    pos = showtext(win, (330, 30), 'Keyboard Focus', (255, 255, 255), bgcolor)
    win.blit(img_on_off[pg.key.get_focused()], pos)
    pos = showtext(win, (10, 60), 'Mouse Position(rel)', (255, 255, 255), bgcolor)
    rel = pg.mouse.get_rel()
    virtual_x += rel[0]
    virtual_y += rel[1]
    mouse_data = tuple(list(pg.mouse.get_pos()) + list(rel))
    p = '%s, %s (%s, %s)' % mouse_data
    showtext(win, pos, p, bgcolor, (255, 255, 55))
    pos = showtext(win, (330, 60), 'Last Keypress', (255, 255, 255), bgcolor)
    if last_key:
        p = '%d, %s' % (last_key, pg.key.name(last_key))
    else:
        p = 'None'
    showtext(win, pos, p, bgcolor, (255, 255, 55))
    pos = showtext(win, (10, 90), 'Input Grabbed', (255, 255, 255), bgcolor)
    win.blit(img_on_off[pg.event.get_grab()], pos)
    is_virtual_mouse = pg.event.get_grab() and (not pg.mouse.get_visible())
    pos = showtext(win, (330, 90), 'Virtual Mouse', (255, 255, 255), bgcolor)
    win.blit(img_on_off[is_virtual_mouse], pos)
    if is_virtual_mouse:
        p = f'{virtual_x}, {virtual_y}'
        showtext(win, (pos[0] + 50, pos[1]), p, bgcolor, (255, 255, 55))