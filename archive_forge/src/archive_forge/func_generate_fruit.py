from pygame.math import Vector2
from Constants import BANNER_HEIGHT, NO_OF_CELLS, USER_SEED
import random
def generate_fruit(self):
    border = NO_OF_CELLS - 1
    x = random.randrange(1, border)
    y = random.randrange(BANNER_HEIGHT, border)
    self.position = Vector2(x, y)