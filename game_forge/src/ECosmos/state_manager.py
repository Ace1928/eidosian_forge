"""
State management for the ECosmos system.

Provides functionality for saving and loading the simulation state,
enabling interruption resistance and state persistence.
"""

import os
import pickle
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from data_structures import Cell, RuleSpecies, Position
import config
from eidosian_core import eidosian

# Set up logging
logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "ecosmos_state.pkl"


class StateManager:
    """
    Handles saving and loading the simulation state.
    """

    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize the state manager.

        Args:
            state_dir: Directory to store state files
        """
        # First try user-specified directory, then config, then fallback to current directory
        if state_dir is not None:
            self.state_dir = state_dir
        else:
            self.state_dir = config.STATE_DIRECTORY

        # Make sure directory exists
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            logger.info(f"Using state directory: {self.state_dir}")
        except Exception as e:
            logger.error(f"Failed to create state directory {self.state_dir}: {e}")
            # Fall back to current directory if there's an issue
            self.state_dir = "."
            logger.info(f"Falling back to current directory for state files")

    @eidosian()
    def get_state_path(self, filename: Optional[str] = None) -> str:
        """
        Get the full path for a state file.

        Args:
            filename: Optional specific filename

        Returns:
            Full path to the state file
        """
        if filename is None:
            filename = DEFAULT_STATE_FILE
        return os.path.join(self.state_dir, filename)

    @eidosian()
    def save_state(
        self,
        world: List[List[Cell]],
        tick: int,
        next_species_id: int,
        stats: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """
        Save the current simulation state to a file.

        Args:
            world: The 2D world grid
            tick: Current simulation tick
            next_species_id: Next available species ID
            stats: Current simulation statistics
            filename: Optional custom filename

        Returns:
            Path to the saved state file
        """
        state_path = self.get_state_path(filename)

        # Create state object
        state = {
            "world": world,
            "tick": tick,
            "next_species_id": next_species_id,
            "stats": stats,
            "timestamp": time.time(),
            "version": "1.0",
        }

        # Save state using pickle
        try:
            with open(state_path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"State saved to {state_path}")
            return state_path
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to save state: {e}")
            # Try to save to an alternate location as a backup
            backup_path = self.get_state_path(f"backup_{int(time.time())}.pkl")
            try:
                with open(backup_path, "wb") as f:
                    pickle.dump(state, f)
                logger.info(f"Backup state saved to {backup_path}")
                return backup_path
            except (IOError, pickle.PickleError):
                logger.critical("Failed to save backup state")
                return ""

    @eidosian()
    def load_state(self, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load simulation state from a file.

        Args:
            filename: Optional specific filename to load

        Returns:
            Loaded state dictionary or None if loading failed
        """
        state_path = self.get_state_path(filename)

        if not os.path.exists(state_path):
            logger.info(f"No state file found at {state_path}")
            return None

        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)

            logger.info(f"State loaded from {state_path}")
            logger.info(f"Resuming from tick {state['tick']}")

            return state
        except (IOError, pickle.PickleError, KeyError) as e:
            logger.error(f"Failed to load state: {e}")
            return None

    @eidosian()
    def state_exists(self, filename: Optional[str] = None) -> bool:
        """
        Check if a state file exists.

        Args:
            filename: Optional specific filename to check

        Returns:
            True if state file exists, False otherwise
        """
        state_path = self.get_state_path(filename)
        return os.path.exists(state_path)

    @eidosian()
    def list_available_states(self) -> List[Tuple[str, float]]:
        """
        List all available state files with their timestamps.

        Returns:
            List of tuples containing (filename, timestamp)
        """
        states = []
        for filename in os.listdir(self.state_dir):
            if filename.endswith(".pkl"):
                path = os.path.join(self.state_dir, filename)
                try:
                    timestamp = os.path.getmtime(path)
                    states.append((filename, timestamp))
                except OSError:
                    continue

        # Sort by timestamp, newest first
        return sorted(states, key=lambda x: x[1], reverse=True)
