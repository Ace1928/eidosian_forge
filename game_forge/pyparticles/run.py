#!/usr/bin/env python3
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)

from pyparticles.main import main

if __name__ == "__main__":
    main()
