import pygame
def tprint(self, screen, text):
    text_bitmap = self.font.render(text, True, (0, 0, 0))
    screen.blit(text_bitmap, (self.x, self.y))
    self.y += self.line_height