#!/usr/bin/env python3
"""
Gene Particles Simulation Runner

A simple launcher script that properly sets up the Python path
to avoid import errors regardless of where the script is run from.

Usage:
    python run_gene_particles.py

Or make executable and run directly:
    chmod +x run_gene_particles.py
    ./run_gene_particles.py
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the main function from gene_particles
from game_forge.src.gene_particles.gp_main import main

if __name__ == "__main__":
    print("🧬 Starting Gene Particles Simulation...")
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Simulation terminated by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("👋 Simulation ended")
