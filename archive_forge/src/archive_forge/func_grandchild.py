import os
import sys
def grandchild() -> None:
    sys.stdout.write('grandchild started')
    sys.stdout.flush()
    sys.stdin.read()