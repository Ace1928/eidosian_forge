import argparse
import os
import platform
import subprocess
import sys
import time
def is_test_name(f):
    return f.startswith('gyptest') and f.endswith('.py')