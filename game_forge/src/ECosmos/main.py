"""
Main entry point for the ECosmos simulation.

This module initializes and runs the evolving computational ecosystem,
coordinating between all the other modules.
"""

import random
import logging
import time
import signal
import sys
from typing import List, Dict, Tuple, Optional, Set
import argparse

from data_structures import Cell, RuleSpecies, Position
from interpreter import run_species
from evolution import (
    distribute_initial_species,
    mutate_species,
    reproduce_species,
    handle_species_interactions,
    inject_environmental_energy,
)
from visualize import WorldVisualizer
from state_manager import StateManager
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for signal handling
interrupt_received = False
world = None
current_tick = 0
next_species_id = 0
current_stats = {}
state_manager = None


def signal_handler(sig, frame):
    """Handle interrupt signals by setting a flag for graceful shutdown."""
    global interrupt_received
    logger.info(f"Signal {sig} received. Preparing for graceful shutdown...")
    interrupt_received = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_world(width: int, height: int) -> List[List[Cell]]:
    """
    Initialize a 2D world grid of cells.

    Args:
        width: Width of the world
        height: Height of the world

    Returns:
        2D list of Cells
    """
    return [[Cell() for _ in range(width)] for _ in range(height)]


def collect_statistics(world: List[List[Cell]], tick: int) -> Dict[str, any]:
    """
    Collect global statistics about the simulation.

    Args:
        world: 2D grid of cells
        tick: Current time step

    Returns:
        Dictionary of statistics
    """
    # Count occupied cells and total energy
    occupied_cells = 0
    total_energy = 0.0
    total_rules = 0
    max_generation = 0

    species_by_generation: Dict[int, int] = {}

    for y in range(config.WORLD_HEIGHT):
        for x in range(config.WORLD_WIDTH):
            occupant = world[y][x].occupant
            if occupant is not None:
                occupied_cells += 1
                total_energy += occupant.energy
                total_rules += len(occupant.rules)
                max_generation = max(max_generation, occupant.generation)

                # Count species by generation
                gen = occupant.generation
                species_by_generation[gen] = species_by_generation.get(gen, 0) + 1

    # Calculate statistics
    avg_energy = total_energy / occupied_cells if occupied_cells > 0 else 0
    avg_rules = total_rules / occupied_cells if occupied_cells > 0 else 0

    # Get top 3 generations by population
    top_generations = sorted(
        species_by_generation.items(), key=lambda x: x[1], reverse=True
    )[:3]

    return {
        "Tick": tick,
        "Population": occupied_cells,
        "Total Energy": f"{total_energy:.1f}",
        "Avg Energy": f"{avg_energy:.1f}",
        "Avg Rules": f"{avg_rules:.1f}",
        "Max Generation": max_generation,
        "Top Generations": ", ".join([f"Gen {g}: {c}" for g, c in top_generations]),
    }


def process_reproduction(world: List[List[Cell]], next_species_id: int) -> int:
    """
    Process reproduction for all species in the world.

    Args:
        world: 2D grid of cells
        next_species_id: Next available species ID

    Returns:
        Updated next_species_id value
    """
    # First pass: identify species that can reproduce and find empty cells
    reproducers = []
    empty_cells = []

    for y in range(config.WORLD_HEIGHT):
        for x in range(config.WORLD_WIDTH):
            if world[y][x].occupant is not None:
                species = world[y][x].occupant
                if species.can_reproduce():
                    reproducers.append((x, y, species))
            else:
                empty_cells.append((x, y))

    # Second pass: perform reproduction if enough empty cells
    random.shuffle(reproducers)
    random.shuffle(empty_cells)

    for (x, y, species), (empty_x, empty_y) in zip(reproducers, empty_cells):
        if len(empty_cells) == 0:
            break

        child = reproduce_species(species, next_species_id)
        if child is not None:
            child.position = Position(empty_x, empty_y)
            world[empty_y][empty_x].add_occupant(child)
            next_species_id += 1
            empty_cells.remove((empty_x, empty_y))

    return next_species_id


def simulate_world(max_ticks: Optional[int] = None) -> None:
    """
    Run the main simulation loop.

    Args:
        max_ticks: Maximum number of time steps to simulate, or None for infinite
    """
    global world, current_tick, next_species_id, current_stats

    # Initialize random seed for reproducibility
    random.seed(config.RANDOM_SEED)

    # Load existing state or create new world
    loaded_state = state_manager.load_state()
    if loaded_state:
        world = loaded_state["world"]
        current_tick = loaded_state["tick"]
        next_species_id = loaded_state["next_species_id"]
        current_stats = loaded_state["stats"]
        logger.info(f"Resuming simulation from tick {current_tick}")
    else:
        # Create the world
        world = create_world(config.WORLD_WIDTH, config.WORLD_HEIGHT)

        # Initialize world with random species
        next_species_id = distribute_initial_species(world)
        current_tick = 0
        logger.info(f"Initialized new world with {next_species_id} species")

    # Set up visualization
    visualizer = WorldVisualizer(config.WORLD_WIDTH, config.WORLD_HEIGHT)

    try:
        # Main simulation loop
        while (
            max_ticks is None or current_tick < max_ticks
        ) and not interrupt_received:
            logger.info(f"Starting tick {current_tick}")

            try:
                # Process each cell in the world
                for y in range(config.WORLD_HEIGHT):
                    for x in range(config.WORLD_WIDTH):
                        cell = world[y][x]
                        if cell.occupant is not None:
                            species = cell.occupant

                            # Skip processing if species has no rules
                            if not species.rules:
                                logger.warning(
                                    f"Species #{species.species_id} has no rules, removing"
                                )
                                world[y][x].remove_occupant()
                                continue

                            # Reset usage stats
                            species.reset_usage_stats()

                            # Run species rules
                            run_species(species)

                            # Only remove species if completely non-viable
                            if species.energy <= 0 or len(species.rules) == 0:
                                world[y][x].remove_occupant()
                                logger.debug(
                                    f"Species #{species.species_id} died at ({x}, {y})"
                                )
                                continue

                            # Apply mutations
                            mutate_species(species)

                            # Handle interactions with environment and neighbors
                            handle_species_interactions(world, x, y)

                # Environmental energy injection for longevity
                inject_environmental_energy(world, current_tick)

                # Process reproduction
                next_species_id = process_reproduction(world, next_species_id)

                # Collect statistics
                current_stats = collect_statistics(world, current_tick)
                logger.info(f"Tick {current_tick} stats: {current_stats}")

                # Periodic state saving
                if current_tick % config.STATE_SAVE_INTERVAL == 0:
                    state_manager.save_state(
                        world, current_tick, next_species_id, current_stats
                    )

                # Update visualization
                if current_tick % config.TICKS_PER_VISUAL_UPDATE == 0:
                    if config.VISUALIZATION_ENABLED:
                        visualizer.update(world, current_tick, current_stats)

                # Check if simulation should continue - only stop if absolutely no species
                if current_stats["Population"] == 0:
                    # Attempt species resurrection if world is empty
                    if current_tick % 50 == 0:  # Every 50 ticks if world is empty
                        logger.info(
                            "World empty, attempting to introduce new species..."
                        )
                        new_count = distribute_initial_species(
                            world, next_species_id, max_new=5
                        )
                        next_species_id += new_count
                        if new_count > 0:
                            logger.info(
                                f"Added {new_count} new species to prevent extinction"
                            )

                current_tick += 1

            except Exception as e:
                logger.error(f"Error in simulation cycle: {e}", exc_info=True)
                # Save state on error for recovery
                state_manager.save_state(
                    world,
                    current_tick,
                    next_species_id,
                    current_stats,
                    "error_recovery.pkl",
                )
                # Continue simulation despite errors
                current_tick += 1
                time.sleep(1)  # Brief pause after error

    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        # Save state on error for recovery
        state_manager.save_state(
            world, current_tick, next_species_id, current_stats, "error_recovery.pkl"
        )
        raise

    finally:
        # Save final state
        if interrupt_received:
            logger.info("Interrupt received. Saving state before exit...")
        else:
            logger.info("Simulation complete. Saving final state...")

        state_manager.save_state(world, current_tick, next_species_id, current_stats)

        logger.info("Simulation ended")

        # Keep visualization window open until closed
        if config.VISUALIZATION_ENABLED:
            logger.info("Close visualization window to exit")
            plt.ioff()
            plt.show()


def main():
    """
    Main entry point with command-line argument parsing.
    """
    global state_manager

    parser = argparse.ArgumentParser(
        description="ECosmos: Evolving Computational Ecosystem"
    )

    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Number of simulation ticks to run (default: infinite)",
    )

    parser.add_argument(
        "--population",
        type=int,
        default=config.INITIAL_POPULATION,
        help=f"Initial population size for new simulations (default: {config.INITIAL_POPULATION})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {config.RANDOM_SEED})",
    )

    parser.add_argument(
        "--no-visual", action="store_true", help="Disable visualization"
    )

    parser.add_argument(
        "--save-frames", action="store_true", help="Save visualization frames to disk"
    )

    parser.add_argument(
        "--new-simulation",
        action="store_true",
        help="Force start a new simulation, ignoring saved state",
    )

    parser.add_argument(
        "--state-dir", type=str, default=None, help="Directory for state files"
    )

    parser.add_argument(
        "--load-state",
        type=str,
        default=None,
        help="Specific state file to load (filename only)",
    )

    args = parser.parse_args()

    # Update config based on command-line arguments
    config.INITIAL_POPULATION = args.population
    config.RANDOM_SEED = args.seed
    config.VISUALIZATION_ENABLED = not args.no_visual
    config.SAVE_FRAMES = args.save_frames

    # Initialize state manager
    state_manager = StateManager(args.state_dir)

    # If loading specific state was requested
    if args.load_state and not args.new_simulation:
        if not state_manager.state_exists(args.load_state):
            logger.error(f"State file {args.load_state} not found")
            sys.exit(1)

    # If forced new simulation, remove existing state
    if args.new_simulation:
        logger.info("Forcing new simulation, ignoring saved state")

    # Run simulation
    simulate_world(max_ticks=args.ticks)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
