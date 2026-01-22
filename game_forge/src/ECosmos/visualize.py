"""
Visualization for the ECosmos system.

This module handles the graphical representation of the evolving world,
showing species distributions, energy levels, and other metrics.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.colors as mcolors

from data_structures import Cell, RuleSpecies
import config

# Set up logging
logger = logging.getLogger(__name__)


class WorldVisualizer:
    """
    Class for visualizing the evolving world.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize the visualization system.

        Args:
            width: World width
            height: World height
        """
        self.width = width
        self.height = height
        self.fig = None
        self.ax = None
        self.im = None
        self.stats_text = None
        self.frame_count = 0

        # Only initialize matplotlib components if visualization is enabled
        if config.VISUALIZATION_ENABLED:
            try:
                self.fig, self.ax = plt.subplots(figsize=(10, 8))

                # Create empty world data array (RGB)
                self.data = np.zeros((self.height, self.width, 3))

                # Initialize the plot
                self.im = self.ax.imshow(self.data, vmin=0.0, vmax=1.0)
                self.ax.set_title("Evolving Computational Cosmos")

                # Add colorbar to show energy levels
                self.energy_sm = plt.cm.ScalarMappable(
                    cmap=plt.cm.viridis,
                    norm=mcolors.Normalize(vmin=0, vmax=config.INITIAL_ENERGY * 1.5),
                )
                self.fig.colorbar(
                    self.energy_sm,
                    ax=self.ax,
                    orientation="vertical",
                    label="Energy Level",
                )

                # Statistics text display
                self.stats_text = self.ax.text(
                    0.02,
                    0.95,
                    "",
                    transform=self.ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )

                # Create output directory if saving frames
                if config.SAVE_FRAMES:
                    os.makedirs(config.OUTPUT_DIRECTORY, exist_ok=True)

                # Show the plot
                plt.ion()  # Turn on interactive mode
                plt.show(block=False)  # Fixed: Don't block here

            except Exception as e:
                logger.error(f"Failed to initialize visualization: {e}")
                config.VISUALIZATION_ENABLED = False
                logger.info("Continuing without visualization")

    def update(self, world: List[List[Cell]], tick: int, stats: Dict[str, any]) -> None:
        """
        Update the visualization with the current world state.

        Args:
            world: 2D grid of cells
            tick: Current time step
            stats: Dictionary of simulation statistics
        """
        # Skip if visualization is disabled
        if not config.VISUALIZATION_ENABLED or self.fig is None:
            return

        try:
            # Reset data array
            self.data = np.zeros((self.height, self.width, 3))

            # Fill with species colors
            energy_data = np.zeros((self.height, self.width))

            for y in range(self.height):
                for x in range(self.width):
                    occupant = world[y][x].occupant
                    if occupant is not None:
                        self.data[y, x] = occupant.color
                        energy_data[y, x] = occupant.energy

            # Update the image data
            self.im.set_data(self.data)
            self.ax.set_title(f"Evolving Computational Cosmos - Tick {tick}")

            # Update statistics text
            stats_str = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            self.stats_text.set_text(stats_str)

            # Save frame if enabled
            if config.SAVE_FRAMES:
                frame_filename = os.path.join(
                    config.OUTPUT_DIRECTORY, f"frame_{self.frame_count:04d}.png"
                )
                plt.savefig(frame_filename, dpi=100)
                self.frame_count += 1

            # Update display
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            # Disable visualization on error
            config.VISUALIZATION_ENABLED = False
            logger.info("Continuing without visualization")

    def show_species_details(self, species: Optional[RuleSpecies]) -> None:
        """
        Display detailed information about a selected species.

        Args:
            species: The species to display details for
        """
        if species is None:
            return

        # Create a new figure for species details
        plt.figure(figsize=(10, 6))

        # Display basic info
        plt.subplot(1, 2, 1)
        info_text = f"Species ID: {species.species_id}\n"
        info_text += f"Generation: {species.generation}\n"
        info_text += f"Energy: {species.energy:.2f}\n"
        info_text += f"Rules: {len(species.rules)}\n"
        info_text += f"CPU Usage: {species.cpu_usage}\n"
        info_text += f"RAM Usage: {species.ram_usage}\n"

        plt.text(0.1, 0.5, info_text, fontsize=12)
        plt.axis("off")

        # Display rules
        plt.subplot(1, 2, 2)
        rules_text = "Rules:\n"
        for i, rule in enumerate(species.rules[:10]):  # Show first 10 rules
            rules_text += f"{i}: {rule.opcode} - {rule.operands}\n"

        if len(species.rules) > 10:
            rules_text += f"...and {len(species.rules) - 10} more rules"

        plt.text(0.1, 0.5, rules_text, fontsize=10)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def close(self) -> None:
        """Close the visualization."""
        if self.fig is not None:
            try:
                plt.ioff()
                plt.close(self.fig)
            except:
                pass
