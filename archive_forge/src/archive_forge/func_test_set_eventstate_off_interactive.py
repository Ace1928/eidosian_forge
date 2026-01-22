import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_set_eventstate_off_interactive(self):
    c = self._get_first_controller()
    if not c:
        self.skipTest('No controller connected')
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((400, 400))
    font = pygame.font.Font(None, 20)
    running = True
    screen.fill((255, 255, 255))
    screen.blit(font.render("Press button 'x' (on ps4) or 'a' (on xbox).", True, (0, 0, 0)), (0, 0))
    pygame.display.update()
    controller.set_eventstate(False)
    while running:
        for event in pygame.event.get(pygame.QUIT):
            if event:
                running = False
        if c.get_button(pygame.CONTROLLER_BUTTON_A):
            if pygame.event.peek(pygame.CONTROLLERBUTTONDOWN):
                pygame.display.quit()
                pygame.font.quit()
                self.fail()
            else:
                running = False
    pygame.display.quit()
    pygame.font.quit()