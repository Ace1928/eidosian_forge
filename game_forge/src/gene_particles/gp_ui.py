#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gene Particles GUI overlay and interaction controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pygame
import numpy as np

from game_forge.src.gene_particles.gp_config import SimulationConfig


ColorRGBA = Tuple[int, int, int, int]


@dataclass
class UIState:
    """Runtime UI state toggles."""

    paused: bool = False
    show_help: bool = True
    show_config: bool = True
    show_stats: bool = True
    single_step: bool = False


@dataclass(frozen=True)
class UITheme:
    """Theme colors for the overlay."""

    panel_bg: ColorRGBA = (12, 16, 24, 200)
    panel_border: ColorRGBA = (60, 120, 160, 220)
    panel_header: ColorRGBA = (24, 32, 48, 220)
    text_primary: Tuple[int, int, int] = (230, 235, 245)
    text_secondary: Tuple[int, int, int] = (170, 185, 210)
    accent: Tuple[int, int, int] = (80, 220, 255)
    warning: Tuple[int, int, int] = (255, 170, 80)


class SimulationUI:
    """GUI overlay for simulation controls and live status."""

    def __init__(self, surface: pygame.Surface, config: SimulationConfig) -> None:
        self.surface = surface
        self.config = config
        self.state = UIState()
        self.theme = UITheme()
        pygame.font.init()
        self.font = pygame.font.SysFont("Fira Sans", 16)
        self.font_small = pygame.font.SysFont("Fira Sans", 14)
        self.font_header = pygame.font.SysFont("Fira Sans", 18, bold=True)

    def handle_event(self, event: pygame.event.Event, automata: object) -> None:
        """Handle input events to adjust simulation config."""
        if event.type != pygame.KEYDOWN:
            return

        key = event.key
        if key in (pygame.K_SPACE, pygame.K_p):
            self.state.paused = not self.state.paused
            self.state.single_step = False
        elif key == pygame.K_n:
            if self.state.paused:
                self.state.single_step = True
        elif key == pygame.K_h:
            self.state.show_help = not self.state.show_help
        elif key == pygame.K_g:
            self.state.show_config = not self.state.show_config
        elif key == pygame.K_s:
            self.state.show_stats = not self.state.show_stats
        elif key == pygame.K_b:
            self.config.boundary_mode = (
                "wrap" if self.config.boundary_mode == "reflect" else "reflect"
            )
        elif key == pygame.K_m:
            self.config.projection_mode = (
                "orthographic"
                if self.config.projection_mode == "perspective"
                else "perspective"
            )
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.config.particle_size = min(self.config.particle_size + 0.2, 10.0)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.config.particle_size = max(self.config.particle_size - 0.2, 0.5)
        elif key == pygame.K_LEFTBRACKET:
            self.config.cluster_radius = max(self.config.cluster_radius - 1.0, 1.0)
        elif key == pygame.K_RIGHTBRACKET:
            self.config.cluster_radius = min(self.config.cluster_radius + 1.0, 200.0)
        elif key == pygame.K_COMMA:
            self.config.global_temperature = max(self.config.global_temperature - 0.02, 0.0)
        elif key == pygame.K_PERIOD:
            self.config.global_temperature = min(self.config.global_temperature + 0.02, 2.0)
        elif key == pygame.K_f:
            self.config.use_force_registry = not self.config.use_force_registry
        elif key == pygame.K_y:
            self._toggle_force_family("yukawa", 0.5)
        elif key == pygame.K_l:
            self._toggle_force_family("lennard_jones", 0.5)
        elif key == pygame.K_o:
            self._toggle_force_family("morse", 0.5)
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3):
            mod = getattr(event, "mod", 0)
            delta = 0.1 if (mod & pygame.KMOD_SHIFT) else -0.1
            if key == pygame.K_1:
                self._adjust_force_family("yukawa", delta)
            elif key == pygame.K_2:
                self._adjust_force_family("lennard_jones", delta)
            else:
                self._adjust_force_family("morse", delta)

    def _toggle_force_family(self, name: str, enabled_value: float) -> None:
        """Toggle a force family scale between zero and a provided value."""
        scale = self.config.force_registry_family_scale
        current = float(scale.get(name, 0.0))
        scale[name] = 0.0 if current != 0.0 else float(enabled_value)

    def _adjust_force_family(self, name: str, delta: float) -> None:
        """Adjust a force family scale by a delta within [0, 2]."""
        scale = self.config.force_registry_family_scale
        current = float(scale.get(name, 0.0))
        scale[name] = float(np.clip(current + delta, 0.0, 2.0))

    def draw_panel(
        self, rect: pygame.Rect, title: str, lines: Iterable[str]
    ) -> None:
        """Draw a themed panel with title and lines."""
        panel = pygame.Surface(rect.size, flags=pygame.SRCALPHA)
        panel.fill(self.theme.panel_bg)
        header_rect = pygame.Rect(0, 0, rect.width, 26)
        pygame.draw.rect(panel, self.theme.panel_header, header_rect, border_radius=6)
        pygame.draw.rect(panel, self.theme.panel_border, panel.get_rect(), width=1, border_radius=6)

        title_surf = self.font_header.render(title, True, self.theme.accent)
        panel.blit(title_surf, (10, 4))

        y = 30
        for line in lines:
            text = self.font_small.render(line, True, self.theme.text_secondary)
            panel.blit(text, (10, y))
            y += 18

        self.surface.blit(panel, rect.topleft)

    def render(self, stats: Dict[str, float]) -> None:
        """Render overlay panels."""
        width, height = self.surface.get_size()

        if self.state.show_stats:
            stats_lines = [
                f"FPS: {stats.get('fps', 0.0):.1f}",
                f"Species: {int(stats.get('total_species', 0))}",
                f"Particles: {int(stats.get('total_particles', 0))}",
                f"Dim: {self.config.spatial_dimensions}D",
                f"Boundary: {self.config.boundary_mode}",
                f"Reproduction: {self.config.reproduction_mode.value}",
            ]
            self.draw_panel(
                pygame.Rect(12, height - 160, 260, 140),
                "Status",
                stats_lines,
            )

        if self.state.show_config:
            config_lines = [
                f"Projection: {self.config.projection_mode}",
                f"Particle size: {self.config.particle_size:.1f}",
                f"Cluster radius: {self.config.cluster_radius:.1f}",
                f"Temperature: {self.config.global_temperature:.2f}",
                f"Depth fade: {self.config.depth_fade_strength:.2f}",
                f"Force registry: {'on' if self.config.use_force_registry else 'off'}",
                (
                    "Force scales Y/L/M: "
                    f"{self.config.force_registry_family_scale.get('yukawa', 0.0):.2f}/"
                    f"{self.config.force_registry_family_scale.get('lennard_jones', 0.0):.2f}/"
                    f"{self.config.force_registry_family_scale.get('morse', 0.0):.2f}"
                ),
            ]
            self.draw_panel(
                pygame.Rect(width - 320, 12, 308, 168),
                "Config",
                config_lines,
            )

        if self.state.show_help:
            help_lines = [
                "Space/P: pause",
                "N: step (paused)",
                "H: toggle help",
                "G: toggle config",
                "S: toggle stats",
                "B: boundary wrap/reflect",
                "M: projection mode",
                "F: force registry toggle",
                "Y/L/O: toggle Yukawa/LJ/Morse",
                "1/2/3: dec Yukawa/LJ/Morse",
                "Shift+1/2/3: inc Yukawa/LJ/Morse",
                "+/-: particle size",
                "[/]: cluster radius",
                ",/.: temperature",
            ]
            self.draw_panel(
                pygame.Rect(width - 320, height - 280, 308, 260),
                "Controls",
                help_lines,
            )
