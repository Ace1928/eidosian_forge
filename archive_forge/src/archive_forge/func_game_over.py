import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def game_over(self):
    again = False
    while not again:
        for event in pygame.event.get():
            if self.is_quit(event):
                again = True
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    again = True
                    break
                if event.key == pygame.K_s:
                    again = True
                    self.controller.save_model()
                    break
        self.display.fill(MENU_COLOR)
        if self.curr_menu.state == 'GA' and self.controller.model_loaded == False:
            best_score = self.controller.best_GA_score()
            best_gen = self.controller.best_GA_gen()
            high_score = f'Best snake Score: {best_score} in generation {best_gen}'
            save = 'Press S to save best snake'
            self.draw_text(save, size=30, x=self.SIZE / 2, y=self.SIZE / 2 + 3 * CELL_SIZE, color=FRUIT_COLOR)
        else:
            high_score = f'High Score: {self.controller.get_score()}'
        to_continue = 'Enter to Continue'
        self.draw_text(high_score, size=35, x=self.SIZE / 2, y=self.SIZE / 2)
        self.draw_text(to_continue, size=30, x=self.SIZE / 2, y=self.SIZE / 2 + 2 * CELL_SIZE, color=WHITE)
        self.window.blit(self.display, (0, 0))
        pygame.display.update()
    self.controller.reset()