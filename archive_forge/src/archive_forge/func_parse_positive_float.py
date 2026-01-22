import argparse
import ast
import re
import sys
def parse_positive_float(value_str):
    value = float(value_str)
    if value < 0:
        raise ValueError('Invalid time %s. Time value must be positive.' % value_str)
    return value