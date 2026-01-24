#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeneParticles: Cellular Automata with Gene Expression and Emergent Behaviors.

A particle simulation modeling cellular entities with genetic traits, adaptive behaviors,
and complex interaction networks.

Features
--------
1. Dynamic Gene Expression & Complex Heredity
    - Mutable traits influencing behavior and reproduction
    - Hierarchical gene clusters with layered mutation strategies
    - Nonlinear genotype-to-phenotype mappings with environmental feedback

2. Adaptive Population Management
    - Real-time performance monitoring with optimization triggers
    - Fitness-based population control
    - Resource-dependent population dynamics

3. Evolutionary Mechanisms & Speciation
    - Resource competition with trait-influenced survival
    - Genetic drift triggering speciation events
    - Complete lineage tracking and phylogenetic trees

4. Multi-Scale Interactions
    - Force-based dynamics (potential, gravitational, synergistic)
    - Species-dependent energy transfer
    - Emergent collective behaviors

5. Vectorized Performance
    - NumPy-based computation
    - Spatial partitioning with KD-trees
    - Adaptive rendering

Requirements
-----------
- Python 3.8+
- NumPy >= 1.20.0
- Pygame >= 2.0.0
- SciPy >= 1.7.0

Installation
-----------
.. code-block:: bash

     pip install numpy pygame scipy

Usage
-----
.. code-block:: bash

     python geneparticles.py

Controls
--------
- ESC: Exit simulation
"""

import os
import sys
from typing import NoReturn
from eidosian_core import eidosian

# Add proper path handling to avoid import issues when running from different locations
script_dir: str = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))

# Use absolute imports to avoid circular dependency issues
from game_forge.src.gene_particles.gp_automata import CellularAutomata
from game_forge.src.gene_particles.gp_config import SimulationConfig


@eidosian()
def main() -> NoReturn:
    """Initialize and run the Gene Particles simulation.

    Creates a simulation with default configuration and executes the main
    evolutionary loop until termination conditions are met.

    The simulation continues until one of the following occurs:
         - User presses the ESC key
         - User closes the simulation window
         - Simulation reaches configured maximum frame count

    Returns
    -------
    NoReturn
         Function terminates via system exit

    Raises
    ------
    ImportError
         If required dependencies are not installed
    RuntimeError
         If simulation initialization fails
    """
    # Create simulation configuration with default parameters
    config: SimulationConfig = SimulationConfig()

    # Initialize the cellular automata system with configuration
    cellular_automata: CellularAutomata = CellularAutomata(config)

    # Execute main simulation loop until termination
    cellular_automata.main_loop()

    # Unreachable but satisfies return type requirements
    sys.exit(0)


if __name__ == "__main__":
    # Entry point when executed directly
    main()
