import logging
import re
def rtr(version, range_, loose):
    return outside(version, range_, '>', loose)