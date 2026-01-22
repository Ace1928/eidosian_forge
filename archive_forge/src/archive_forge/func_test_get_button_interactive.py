import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_get_button_interactive(self):
    c = self._get_first_controller()
    if not c:
        self.skipTest('No controller connected')
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((400, 400))
    font = pygame.font.Font(None, 20)
    running = True
    label1 = font.render("Press button 'x' (on ps4) or 'a' (on xbox).", True, (0, 0, 0))
    label2 = font.render('The two values should match up. Press "y" or "n" to confirm.', True, (0, 0, 0))
    is_pressed = [False, False]
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.CONTROLLERBUTTONDOWN and event.button == 0:
                is_pressed[0] = True
            if event.type == pygame.CONTROLLERBUTTONUP and event.button == 0:
                is_pressed[0] = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    running = False
                if event.key == pygame.K_n:
                    running = False
                    pygame.display.quit()
                    pygame.font.quit()
                    self.fail()
        is_pressed[1] = c.get_button(pygame.CONTROLLER_BUTTON_A)
        screen.fill((255, 255, 255))
        screen.blit(label1, (0, 0))
        screen.blit(label2, (0, 20))
        screen.blit(font.render(str(is_pressed), True, (0, 0, 0)), (0, 40))
        pygame.display.update()
    pygame.display.quit()
    pygame.font.quit()