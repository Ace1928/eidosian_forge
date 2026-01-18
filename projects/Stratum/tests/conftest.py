"""
Pytest configuration and fixtures for Stratum tests.

This module sets up the Python path to allow imports of the Stratum
package modules without requiring a package installation.
"""

import sys
import os

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
