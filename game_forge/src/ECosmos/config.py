"""
🌍 Global Configuration - ECosmos Computational Ecosystem 🌍

This module centralizes all simulation parameters to enable easy experimentation
with different settings and scenarios. Each section is organized by functional area
and annotated with detailed explanations.
"""

from typing import Dict, Final, NamedTuple, List, Tuple, Optional, Union
import os

# 🎮 ---------------------
# 🎮 Simulation Settings
# 🎮 ---------------------
WORLD_WIDTH: Final[int] = 100  # Width of the simulation grid
WORLD_HEIGHT: Final[int] = 100  # Height of the simulation grid
INITIAL_POPULATION: Final[int] = (
    800  # Reduced to give more room for growth and reduce early competition
)
MAX_TIME_STEPS: Final[Optional[int]] = (
    None  # None for indefinite running, int for limited steps
)
TICKS_PER_VISUAL_UPDATE: Final[int] = 1  # Update visualization every N ticks
RANDOM_SEED: Final[int] = (
    42  # Seed for random number generation, ensures reproducibility
)
STATE_SAVE_INTERVAL: Final[int] = 10  # Save simulation state every N ticks

# 🧬 ---------------------
# 🧬 Evolution Parameters
# 🧬 ---------------------
MUTATION_RATE: Final[float] = 0.08  # Increased to promote speciation and innovation
RECOMBINATION_RATE: Final[float] = 0.12  # Increased to promote genetic diversity
ENERGY_TRANSFER_FRACTION: Final[float] = (
    0.15  # Increased for more efficient energy sharing
)
ENERGY_DECAY_RATE: Final[float] = 0.003  # Reduced to make survival less punishing
INITIAL_ENERGY: Final[float] = 120.0  # Increased to give new species a better start
MAX_ENERGY: Final[float] = (
    300.0  # Increased to allow energy storage for reproduction events
)
MIN_VIABLE_ENERGY: Final[float] = 3.0  # Lowered to increase survival chances
REPRODUCTION_THRESHOLD: Final[float] = (
    70.0  # Lowered to make reproduction more accessible
)
REPRODUCTION_ENERGY_COST: Final[float] = (
    40.0  # Reduced to encourage more reproduction events
)
RULE_ADDITION_CHANCE: Final[float] = (
    0.25  # Increased to promote development of novel behaviors
)
RULE_DELETION_CHANCE: Final[float] = (
    0.04  # Slightly reduced to preserve successful rule sets
)
GENE_TRANSFER_CHANCE: Final[float] = (
    0.15  # Increased to promote horizontal gene transfer
)
ENVIRONMENTAL_ENERGY_FACTOR: Final[float] = (
    3.0  # Increased to provide more energy input to the system
)

# ⚡ Energy Management
BASE_ENERGY_CONSUMPTION: Final[float] = 0.015  # Reduced to make survival easier
RULE_EXECUTION_COST: Final[float] = 0.008  # Reduced to encourage more rule execution

# ⏱️ ---------------------
# ⏱️ Execution Constraints
# ⏱️ ---------------------
TIMEOUT_LIMIT: Final[int] = 25  # Increased to allow more complex rule execution
CPU_BUDGET: Final[int] = 180_000  # Increased to allow more computation
RAM_BUDGET: Final[int] = 350_000  # Increased to allow more memory usage
MAX_RULES_PER_SPECIES: Final[int] = 25  # Increased to allow more complex behaviors
MIN_RULES_FOR_VIABILITY: Final[int] = (
    2  # Reduced to increase viability of simpler species
)
MAX_RULE_LENGTH: Final[int] = 60  # Increased to allow more complex rules
INITIAL_MIN_RULES: Final[int] = 3  # Kept the same for minimum viability
INITIAL_MAX_RULES: Final[int] = (
    10  # Increased to give new species more behavioral options
)
MAX_CYCLES_PER_TICK: Final[int] = 120  # Increased to allow more rule execution per tick
INSTRUCTION_TIMEOUT_MS: Final[int] = (
    6  # Slightly increased for more complex instructions
)

# 🎨 ---------------------
# 🎨 Visualization Settings
# 🎨 ---------------------
COLOR_MUTATION_CHANCE: Final[float] = 0.02  # Probability of color mutation per tick
ENERGY_COLOR_INFLUENCE: Final[float] = 0.3  # How much energy affects species brightness
VISUALIZATION_ENABLED: Final[bool] = True  # Toggle for visualization system
SAVE_FRAMES: Final[bool] = False  # Whether to save visualization frames to disk
CELL_SIZE: Final[int] = 8  # Pixel size of each cell in visualization
BACKGROUND_COLOR: Final[Tuple[int, int, int]] = (
    0,
    0,
    0,
)  # RGB background color (black)
EMPTY_COLOR: Final[Tuple[int, int, int]] = (
    20,
    20,
    20,
)  # RGB color for empty cells (dark gray)
COLOR_MAPPING: Final[Dict[str, Dict[str, Tuple[int, int, int]]]] = {
    "rule_count": {
        "min": (50, 50, 255),  # Blue for fewer rules
        "max": (255, 50, 50),  # Red for more rules
    },
    "energy": {
        "min": (20, 100, 20),  # Darker green for less energy
        "max": (50, 255, 50),  # Brighter green for more energy
    },
}

# 📁 ---------------------
# 📁 File Paths & Directories
# 📁 ---------------------
OUTPUT_DIRECTORY: Final[str] = os.path.join(
    os.path.dirname(__file__), "output"
)  # For general outputs
STATE_DIRECTORY: Final[str] = os.path.join(
    os.path.dirname(__file__), "state"
)  # For simulation states
STATE_DIR: Final[str] = "states"  # Subdirectory name for simulation states
FRAME_DIR: Final[str] = "frames"  # Subdirectory name for visualization frames


# 🌱 ---------------------
# 🌱 Environment Factors
# 🌱 ---------------------
class EnvironmentFactor(NamedTuple):
    """
    Represents an environmental condition that affects species behavior and evolution.

    Attributes:
        name: Identifier for this environmental factor
        influence_strength: How strongly this factor affects species (0.0 to 1.0)
        distribution: Spatial pattern of this factor ("uniform", "gradient", "radial", "patches")
    """

    name: str
    influence_strength: float  # 0.0 to 1.0
    distribution: str  # 'uniform', 'gradient', 'radial', 'patches'


ENVIRONMENT_FACTORS: Final[List[EnvironmentFactor]] = [
    EnvironmentFactor(
        "temperature", 0.6, "gradient"
    ),  # Increased influence of temperature
    EnvironmentFactor("radiation", 0.4, "radial"),  # Increased influence of radiation
    EnvironmentFactor(
        "resources", 0.85, "patches"
    ),  # Significantly increased importance of resources
    EnvironmentFactor(
        "moisture", 0.7, "gradient"
    ),  # Added new factor to create more niches
    EnvironmentFactor(
        "nutrients", 0.5, "patches"
    ),  # Added new factor to increase resource diversity
]

# 📊 ---------------------
# 📊 Instruction Costs
# 📊 ---------------------
# Energy costs for each operation type (aligned with OperationType enum in data_structures.py)
INSTRUCTION_COSTS: Final[Dict[str, float]] = {
    # 🏃 Movement and sensing - reduced to encourage exploration
    "MOVE": 1.5,  # Reduced cost to encourage mobility
    "SENSE": 0.7,  # Reduced to encourage environmental sensing
    "CONSUME": 2.0,  # Slightly reduced to encourage resource consumption
    # 🧠 Memory operations - mostly unchanged as they're already efficient
    "STORE": 0.8,  # Slightly reduced to encourage memory usage
    "LOAD": 0.8,  # Slightly reduced to encourage memory usage
    # 🧮 Computational operations
    "CALCULATE": 1.2,  # Reduced to encourage calculations
    # 🔀 Flow control - reduced to encourage complex behaviors
    "BRANCH": 0.8,  # Reduced to encourage decision making
    "JUMP": 0.8,  # Reduced to encourage complex rule execution
    # 🔄 Reproduction and interaction - balanced for ecosystem stability
    "REPRODUCE": 4.0,  # Reduced to encourage reproduction
    "SHARE": 1.5,  # Reduced to encourage beneficial social behaviors
    "ATTACK": 3.5,  # Slightly increased to discourage excessive predation
    "DEFEND": 1.8,  # Slightly reduced to make defense viable
    # 🧪 Advanced operations
    "MUTATE": 3.0,  # Reduced to encourage self-modification
}
