from kivy.vector import Vector
def x_axis_angle(q):
    if q == P:
        return 10000000000.0
    return abs((q - P).angle((1, 0)))