# // Creating the snake, apple and A* search algorithm
# Creating the snake, apple and search algorithm
# let snake;
import snake

# let apple;
import apple

# let search;
import search

# Rest of the Imports required in alignment with the rest of the classes.
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint


# // Setting up everything
# Initial setup of the game environment
# function setup() {
def setup():
    #    createCanvas(1200, 600);
    pg.init()
    screen = pg.display.set_mode((1200, 600))
    #    snake = new Snake();
    snake = snake.Snake()
    #    apple = new Apple();
    apple = apple.Apple()
    #    search = new Search(snake, apple);
    search = search.Search(snake, apple)
    #    search.getPath();
    search.get_path()
    #    frameRate(200);
    clock = pg.time.Clock()
    return screen, snake, apple, search, clock


# }

# function draw() {

#    background(51);

#    snake.show();

#    apple.show();

#    snake.update(apple);

# }
