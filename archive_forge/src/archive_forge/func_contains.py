import argparse
import ast
import re
import sys
def contains(self, value):
    if value < self.start or (value == self.start and (not self.start_included)):
        return False
    if value > self.end or (value == self.end and (not self.end_included)):
        return False
    return True