from time import time
from kivy.config import Config
from kivy.vector import Vector
Find a triple tap touch within *self.touches*.
        The touch must be not be a previous triple tap and the distance
        must be within the bounds specified. Additionally, the touch profile
        must be the same kind of touch.
        