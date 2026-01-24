#!/usr/bin/env python3
"""System context utilities for glyph_stream."""
from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict
from eidosian_core import eidosian


class SystemContext:
    """Self-aware execution environment with adaptive capabilities."""

    def __init__(self, check_network: bool = False) -> None:
        self.check_network = check_network
        self.attributes = self._gather_system_attributes()
        self.capabilities = self._analyze_capabilities()
        self.constraints = self._detect_constraints()
        self.optimization_paths = self._generate_optimization_paths()

    def _gather_system_attributes(self) -> Dict[str, Any]:
        attributes = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp": datetime.now().isoformat(),
        }

        try:
            attributes.update(
                {
                    "terminal_width": os.get_terminal_size().columns,
                    "terminal_height": os.get_terminal_size().lines,
                    "is_interactive": sys.stdout.isatty(),
                    "supports_unicode": sys.stdout.encoding.lower().startswith(
                        ("utf", "latin")
                    ),
                    "supports_ansi_color": self._detect_color_support(attributes),
                }
            )
        except (AttributeError, OSError):
            attributes.update(
                {
                    "terminal_width": 80,
                    "terminal_height": 24,
                    "is_interactive": False,
                    "supports_unicode": True,
                    "supports_ansi_color": False,
                }
            )

        try:
            import psutil

            memory = psutil.virtual_memory()
            attributes.update(
                {
                    "memory_total": memory.total,
                    "memory_available": memory.available,
                    "memory_percent": memory.percent,
                    "cpu_count": psutil.cpu_count(logical=True),
                    "cpu_physical": psutil.cpu_count(logical=False),
                }
            )
        except ImportError:
            attributes["memory_detection"] = "unavailable"

        if self.check_network:
            attributes["network_connected"] = self._check_network_connection()
        else:
            attributes["network_connected"] = False

        attributes.update(self._detect_hardware_acceleration(attributes))
        return attributes

    def _detect_hardware_acceleration(self, attrs: Dict[str, Any]) -> Dict[str, bool]:
        acceleration = {
            "numpy_optimized": False,
            "has_mkl": False,
            "has_cuda": False,
            "has_metal": False,
        }

        try:
            import numpy as np

            config_info = np.__config__.show()
            if isinstance(config_info, str):
                acceleration["numpy_optimized"] = (
                    "mkl" in config_info.lower() or "openblas" in config_info.lower()
                )
                acceleration["has_mkl"] = "mkl" in config_info.lower()
        except (ImportError, AttributeError):
            pass

        platform_name = attrs.get("platform", "")
        if os.environ.get("EIDOSIAN_ENABLE_TORCH") == "1":
            try:
                if platform_name != "Darwin":
                    import torch

                    acceleration["has_cuda"] = torch.cuda.is_available()
            except ImportError:
                pass

        if platform_name == "Darwin":
            acceleration["has_metal"] = True

        return acceleration

    def _detect_color_support(self, attrs: Dict[str, Any]) -> bool:
        if "NO_COLOR" in os.environ:
            return False

        if "COLORTERM" in os.environ:
            return True

        platform_name = attrs.get("platform", "")
        if platform_name == "Windows":
            if sys.getwindowsversion().build >= 14931:
                return True
            return any(env_var in os.environ for env_var in ("ANSICON", "ConEmuANSI"))

        term = os.environ.get("TERM", "").lower()
        return term != "dumb" and term.endswith(("color", "ansi", "-256color"))

    def _check_network_connection(self) -> bool:
        try:
            import socket

            socket.create_connection(("8.8.8.8", 53), timeout=1.0)
            return True
        except (socket.error, socket.timeout):
            return False

    def _analyze_capabilities(self) -> Dict[str, bool]:
        return {
            "can_display_unicode": self.attributes.get("supports_unicode", True),
            "can_display_color": self.attributes.get("supports_ansi_color", True),
            "can_display_animations": (
                self.attributes.get("is_interactive", False)
                and self.attributes.get("supports_ansi_color", False)
            ),
            "has_high_performance": self._estimate_performance_tier() >= 2,
            "has_network_access": self.attributes.get("network_connected", False),
        }

    @eidosian()
    def refresh(self, check_network: bool = False) -> None:
        self.check_network = check_network
        self.attributes = self._gather_system_attributes()
        self.capabilities = self._analyze_capabilities()
        self.constraints = self._detect_constraints()
        self.optimization_paths = self._generate_optimization_paths()

    def _detect_constraints(self) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}

        width = self.attributes.get("terminal_width", 80)
        height = self.attributes.get("terminal_height", 24)

        if width < 60:
            constraints["limited_width"] = True
            constraints["max_art_width"] = width - 2
        else:
            constraints["limited_width"] = False
            constraints["max_art_width"] = width - 4

        if height < 20:
            constraints["limited_height"] = True
            constraints["max_art_height"] = height - 4
        else:
            constraints["limited_height"] = False
            constraints["max_art_height"] = height - 6

        perf_tier = self._estimate_performance_tier()
        constraints["performance_tier"] = perf_tier

        if perf_tier == 0:
            constraints["max_scale_factor"] = 1
            constraints["default_fps"] = 5
        elif perf_tier == 1:
            constraints["max_scale_factor"] = 2
            constraints["default_fps"] = 10
        else:
            constraints["max_scale_factor"] = 4
            constraints["default_fps"] = 15

        return constraints

    def _generate_optimization_paths(self) -> Dict[str, Any]:
        paths: Dict[str, Any] = {}

        if self.constraints.get("limited_width", False):
            paths["block_width"] = 4
            paths["block_height"] = 8
        else:
            cpu_physical = self.attributes.get("cpu_physical", 2)
            if cpu_physical >= 8:
                paths["block_width"] = 6
                paths["block_height"] = 6
            else:
                paths["block_width"] = 8
                paths["block_height"] = 8

        if self.constraints.get("performance_tier", 1) <= 0:
            paths["default_edge_mode"] = "simple"
            paths["animation_level"] = 0
        else:
            paths["default_edge_mode"] = "enhanced"
            paths["animation_level"] = (
                2 if self.capabilities.get("has_high_performance", False) else 1
            )

        perf_tier = self.constraints.get("performance_tier", 1)
        paths["ideal_scale_factor"] = min(
            perf_tier + 1, self.constraints.get("max_scale_factor", 2)
        )

        if not self.capabilities.get("can_display_color", True):
            paths["default_output_mode"] = "ascii"
        elif not self.capabilities.get("can_display_unicode", True):
            paths["default_output_mode"] = "ascii-color"
        else:
            paths["default_output_mode"] = "unicode-color"

        return paths

    def _estimate_performance_tier(self) -> int:
        score = 1

        cpu_count = self.attributes.get("cpu_count", 2)
        if cpu_count >= 16:
            score += 2
        elif cpu_count >= 8:
            score += 1
        elif cpu_count <= 2:
            score -= 1

        memory_gb = self.attributes.get("memory_available", 4 * 1024 * 1024 * 1024) / (
            1024 * 1024 * 1024
        )
        if memory_gb >= 16:
            score += 1
        elif memory_gb <= 2:
            score -= 1

        if self.attributes.get("has_cuda", False) or self.attributes.get(
            "has_metal", False
        ):
            score += 1

        return max(0, min(score, 3))

    @eidosian()
    def get_optimized_parameters(self) -> Dict[str, Any]:
        return {
            "scale_factor": self.optimization_paths.get("ideal_scale_factor", 2),
            "block_width": self.optimization_paths.get("block_width", 8),
            "block_height": self.optimization_paths.get("block_height", 8),
            "fps": self.constraints.get("default_fps", 15),
            "edge_mode": self.optimization_paths.get("default_edge_mode", "enhanced"),
            "color_mode": self.capabilities.get("can_display_color", True),
            "animation_level": self.optimization_paths.get("animation_level", 1),
            "max_width": self.constraints.get("max_art_width", 80),
            "max_height": self.constraints.get("max_art_height", 24),
        }

    @eidosian()
    def get_environment_summary(self) -> Dict[str, Any]:
        return {
            "platform": self.attributes.get("platform", "Unknown"),
            "terminal_dimensions": f"{self.attributes.get('terminal_width', 80)}×{self.attributes.get('terminal_height', 24)}",
            "color_support": (
                "Yes" if self.capabilities.get("can_display_color", False) else "No"
            ),
            "unicode_support": (
                "Yes" if self.capabilities.get("can_display_unicode", False) else "No"
            ),
            "animation_support": (
                "Yes"
                if self.capabilities.get("can_display_animations", False)
                else "No"
            ),
            "performance_tier": ["Low", "Medium", "High", "Very High"][
                self.constraints.get("performance_tier", 1)
            ],
            "network_access": (
                "Yes" if self.capabilities.get("has_network_access", False) else "No"
            ),
            "optimization_level": self.optimization_paths.get("animation_level", 1),
        }


SYSTEM_CONTEXT = SystemContext(check_network=False)
SYSTEM_INFO = SYSTEM_CONTEXT.attributes
