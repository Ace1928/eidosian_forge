import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def iwords(text):
    head = 0
    tail = head
    end = len(text)
    while head < end:
        if text[head] == ' ':
            head += 1
            tail = head + 1
        elif text[head] == '\n':
            head += 1
            yield '\n'
            tail = head + 1
        elif tail == end:
            yield text[head:]
            head = end
        elif text[tail] == '\n':
            yield text[head:tail]
            head = tail
        elif text[tail] == ' ':
            yield text[head:tail]
            head = tail
        else:
            tail += 1