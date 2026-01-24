#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gene Particles Rendering System.

A high-performance visualization engine for cellular automata simulations with
energy-modulated aesthetics, statistical displays, and alpha-composited rendering.

The renderer employs optimized techniques including:
    - Vectorized drawing operations for maximum performance
    - Energy-based visual properties for intuitive state comprehension
    - Alpha composition for depth perception and visual clarity
    - Color-coded statistical displays for real-time monitoring

This module transforms simulation state into meaningful visual representation
while maintaining minimal CPU overhead through optimized drawing techniques.
"""

from typing import Dict

import numpy as np
import pygame

from game_forge.src.gene_particles.gp_config import (
from eidosian_core import eidosian
    FPS_COLOR,
    PARTICLES_COLOR,
    SPECIES_COLOR,
    STATS_BG_ALPHA,
    STATS_HEIGHT,
    SimulationConfig,
)
from game_forge.src.gene_particles.gp_types import (
    BoolArray,
    CellularTypeData,
    ColorRGB,
    FloatArray,
    IntArray,
)


class Renderer:
    """High-performance visualization engine for cellular components.

    Handles rendering of particle systems with optimized techniques including
    alpha compositing, energy-modulated coloration, and vectorized operations.
    Creates a multi-layered visual representation that conveys simulation state
    through intuitive visual encoding of energy levels and other properties.

    Attributes:
        surface (pygame.Surface): Main rendering target where all visualization is composited.
        config (SimulationConfig): Configuration parameters controlling rendering properties.
        width (int): Width of the rendering surface in pixels.
        height (int): Height of the rendering surface in pixels.
        particle_surface (pygame.Surface): Alpha-enabled surface for optimized particle drawing.
        stats_surface (pygame.Surface): Semi-transparent surface for statistics overlay.
        font (pygame.font.Font): Font renderer for statistics text.
    """

    def __init__(self, surface: pygame.Surface, config: SimulationConfig) -> None:
        """Initialize the rendering engine with necessary surfaces and resources.

        Creates alpha-enabled composition layers for particles and statistics,
        calculates dimensions, and initializes font rendering capabilities.

        Args:
            surface: The target Pygame surface for all visualization output.
            config: Configuration parameters controlling rendering properties.
        """
        self.surface: pygame.Surface = surface
        self.config: SimulationConfig = config

        # Extract dimensions for convenience and boundary calculations
        self.width: int
        self.height: int
        self.width, self.height = self.surface.get_size()

        # Create alpha-enabled surface for particle rendering with transparency
        self.particle_surface: pygame.Surface = pygame.Surface(
            (self.width, self.height), flags=pygame.SRCALPHA
        ).convert_alpha()

        # Create semi-transparent overlay for statistics display
        self.stats_surface: pygame.Surface = pygame.Surface(
            (self.width, STATS_HEIGHT), flags=pygame.SRCALPHA
        ).convert_alpha()

        # Initialize font renderer for statistics display
        pygame.font.init()
        self.font: pygame.font.Font = pygame.font.SysFont("Arial", 20)

    @eidosian()
    def draw_component(
        self,
        x: float,
        y: float,
        color: ColorRGB,
        energy: float,
        speed_factor: float,
    ) -> None:
        """Draw a single cellular component with energy-modulated visual properties.

        Renders a particle as a circle with color intensity and size that
        visually encode its energy level and speed characteristics.

        Args:
            x: Horizontal position of the component.
            y: Vertical position of the component.
            color: Base RGB color tuple for the component (r,g,b).
            energy: Current energy level affecting brightness (0-100).
            speed_factor: Speed trait affecting color intensity.
        """
        # Clamp energy to valid range and normalize to [0,1]
        health: float = min(100.0, max(0.0, energy))
        intensity_factor: float = health / 100.0

        # Calculate energy and speed modulated color values
        # Higher energy/speed creates brighter particles, lower creates dimmer ones
        r: int = min(
            255,
            int(
                color[0] * intensity_factor * speed_factor
                + (1 - intensity_factor) * 100
            ),
        )
        g: int = min(
            255,
            int(
                color[1] * intensity_factor * speed_factor
                + (1 - intensity_factor) * 100
            ),
        )
        b: int = min(
            255,
            int(
                color[2] * intensity_factor * speed_factor
                + (1 - intensity_factor) * 100
            ),
        )

        # Energy affects particle size for additional visual differentiation
        size_factor: float = 0.8 + (intensity_factor * 0.4)
        particle_size: int = int(self.config.particle_size * size_factor)

        # Pack calculated RGB values into a color tuple
        c: ColorRGB = (r, g, b)

        # Render the component as a circle on the particle surface
        pygame.draw.circle(self.particle_surface, c, (int(x), int(y)), particle_size)

    @eidosian()
    def draw_cellular_type(self, ct: CellularTypeData) -> None:
        """Draw all alive components of a specific cellular type.

        Uses vectorized operations to efficiently identify and render
        only the active components, skipping any dead ones.

        Args:
            ct: Cellular type data containing component properties and state.
        """
        # Extract living component indices for efficient iteration
        alive_array: BoolArray = ct.alive
        alive_indices: IntArray = np.where(alive_array)[0].astype(np.int_)

        # Get typed references to component property arrays
        x_array: FloatArray = ct.x
        y_array: FloatArray = ct.y
        energy_array: FloatArray = ct.energy
        speed_factor_array: FloatArray = ct.speed_factor

        # Draw each living component with its specific properties
        for idx in alive_indices:
            # Extract scalar values from arrays for the component
            x_pos: float = float(x_array[idx])
            y_pos: float = float(y_array[idx])
            energy_val: float = float(energy_array[idx])
            speed_factor_val: float = float(speed_factor_array[idx])

            # Render the component with its extracted properties
            self.draw_component(x_pos, y_pos, ct.color, energy_val, speed_factor_val)

    def _render_statistics(self, stats: Dict[str, float]) -> None:
        """Render simulation statistics with visually distinct styling.

        Creates a semi-transparent overlay with color-coded metrics
        for improved readability and immediate comprehension.

        Args:
            stats: Dictionary of simulation statistics to display
                (fps, total_species, total_particles)
        """
        # Apply semi-transparent dark background for readability
        self.stats_surface.fill((20, 20, 30, STATS_BG_ALPHA))

        # Extract statistics with safe defaults
        fps: float = stats.get("fps", 0.0)
        species_count: int = int(stats.get("total_species", 0))
        particle_count: int = int(stats.get("total_particles", 0))

        # Render each statistic with its distinct color for visual categorization
        fps_text: pygame.Surface = self.font.render(f"FPS: {fps:.1f}", True, FPS_COLOR)
        species_text: pygame.Surface = self.font.render(
            f"Species: {species_count}", True, SPECIES_COLOR
        )
        particles_text: pygame.Surface = self.font.render(
            f"Particles: {particle_count}", True, PARTICLES_COLOR
        )

        # Position and composite each text element onto the stats surface
        self.stats_surface.blit(fps_text, (20, 15))
        self.stats_surface.blit(species_text, (150, 15))
        self.stats_surface.blit(particles_text, (300, 15))

        # Apply the completed stats overlay to the main surface
        self.surface.blit(self.stats_surface, (0, 0))

    @eidosian()
    def render(self, stats: Dict[str, float]) -> None:
        """Render all visualization elements to the main surface.

        Composites multiple visual layers and renders statistical information
        in a single optimized drawing pass.

        Args:
            stats: Dictionary of simulation statistics to display
                (fps, total_species, total_particles)
        """
        # Composite the particle layer onto the main surface
        self.surface.blit(self.particle_surface, (0, 0))

        # Clear the particle surface with full transparency for the next frame
        self.particle_surface.fill((0, 0, 0, 0))

        # Render statistics overlay with enhanced visual styling
        self._render_statistics(stats)
