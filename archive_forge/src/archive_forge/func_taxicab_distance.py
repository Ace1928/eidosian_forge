from kivy.config import Config
def taxicab_distance(self, p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])