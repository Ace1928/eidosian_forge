"""
This module defines the Tile and Player classes for a 2048 game AI. It includes functionality for tile movement,
merging, and player actions such as adding new tiles and moving existing ones based on the game's rules.

Classes:
- Tile: Represents a single tile in the 2048 game, including its position, value, and state.
- Player: Manages the game state, including the tiles on the board, score, and movements.

Dependencies:
- numpy: Used for array manipulation and mathematical operations.
- random: Used for generating random numbers for tile placement and values.
"""

import numpy as np
import random
from typing import List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Tile:
    """
    Represents a tile in the 2048 game with a position, value, and state indicators for merging and movement.

    Attributes:
        position (np.array): The (x, y) position of the tile on the game board.
        value (int): The numerical value of the tile.
        death_on_impact (bool): Flag indicating if the tile will be merged (and thus removed) on the next move.
        already_increased (bool): Flag to prevent a tile from merging more than once per move.
    """

    def __init__(self, x: int, y: int, value: int = 2) -> None:
        """
        Initializes a Tile instance.

        Parameters:
            x (int): The x-coordinate of the tile's position.
            y (int): The y-coordinate of the tile's position.
            value (int, optional): The value of the tile. Defaults to 2.
        """
        self.position = np.array([x, y], dtype=int)
        self.value = value
        self.death_on_impact = False
        self.already_increased = False
        logging.debug(
            f"Tile created at position {self.position} with value {self.value}"
        )

    def move_to(self, new_position: np.array) -> None:
        """
        Updates the tile's position.

        Parameters:
            new_position (np.array): The new position to move the tile to.
        """
        logging.debug(f"Moving tile from {self.position} to {new_position}")
        self.position = new_position

    def set_color(self) -> None:
        """
        Placeholder method to update the tile's color based on its value. Should be implemented in a UI-specific manner.
        """
        pass  # Placeholder for UI integration

    def show(self) -> None:
        """
        Placeholder method to display the tile in the UI. Should be implemented in a UI-specific manner.
        """
        print(
            f"Tile at {self.position} with value {self.value}"
        )  # Placeholder for UI integration

    def clone(self) -> "Tile":
        """
        Creates a copy of the tile.

        Returns:
            Tile: A new Tile instance with the same position and value.
        """
        return Tile(self.position[0], self.position[1], self.value)


class Player:
    """
    Manages the state and behavior of a 2048 game player, including tile positions, movements, and score.

    Attributes:
        fitness (int): A metric for evaluating the player's performance.
        dead (bool): Indicates whether the player has lost the game.
        score (int): The player's current score.
        tiles (List[Tile]): A list of Tile objects representing the tiles on the board.
        empty_positions (List[np.array]): A list of positions on the board that do not contain a tile.
        move_direction (np.array): The current direction in which the tiles are moving.
        moving_tiles (bool): Flag indicating whether the tiles are currently moving.
        tile_moved (bool): Flag indicating whether at least one tile has moved during the last move.
        starting_positions (np.array): The starting positions and values of the initial two tiles.
    """

    def __init__(self, is_replay: bool = False) -> None:
        """
        Initializes a Player instance.

        Parameters:
            is_replay (bool, optional): Indicates whether this player instance is for a replay. Defaults to False.
        """
        self.fitness: int = 0
        self.dead: bool = False
        self.score: int = 0
        self.tiles: List[Tile] = []
        self.empty_positions: List[np.array] = []
        self.move_direction: np.array = np.array([0, 0], dtype=int)
        self.moving_tiles: bool = False
        self.tile_moved: bool = False
        self.starting_positions: np.array = np.zeros((2, 3), dtype=int)

        self.fill_empty_positions()
        if not is_replay:
            self.add_new_tile()
            self.add_new_tile()
            self.set_starting_positions()
        logging.debug("Player initialized")

    def fill_empty_positions(self) -> None:
        self.empty_positions = [np.array([i, j]) for i in range(4) for j in range(4)]

    def set_empty_positions(self) -> None:
        self.empty_positions.clear()
        for i in range(4):
            for j in range(4):
                if self.get_value(i, j) == 0:
                    self.empty_positions.append(np.array([i, j]))

    def set_starting_positions(self) -> None:
        if len(self.tiles) >= 2:
            self.starting_positions[0, :] = np.append(
                self.tiles[0].position, self.tiles[0].value
            )
            self.starting_positions[1, :] = np.append(
                self.tiles[1].position, self.tiles[1].value
            )

    def add_new_tile(self, value: Optional[int] = None) -> None:
        if not self.empty_positions:
            return
        index = random.randint(0, len(self.empty_positions) - 1)
        position = self.empty_positions.pop(index)
        if value is None:
            value = 4 if random.random() < 0.1 else 2
        new_tile = Tile(position[0], position[1], value)
        new_tile.set_color()
        self.tiles.append(new_tile)

    def add_new_tile(self, value: Optional[int] = None) -> None:
        """
        Adds a new tile to the game board at a random empty position.

        Parameters:
            value (Optional[int]): The value of the new tile. If None, the value is randomly set to 2 or 4.
        """
        if not self.empty_positions:
            logging.warning("No empty positions available to add a new tile.")
            return

        try:
            index = random.randint(0, len(self.empty_positions) - 1)
            position = self.empty_positions.pop(index)
            if value is None:
                value = 4 if random.random() < 0.1 else 2
            new_tile = Tile(position[0], position[1], value)
            new_tile.set_color()  # Placeholder for UI integration
            self.tiles.append(new_tile)
            logging.info(f"Added new tile at {position} with value {value}")
        except Exception as e:
            logging.error(f"Failed to add a new tile: {e}")

    def show(self) -> None:
        for tile in sorted(self.tiles, key=lambda x: x.death_on_impact):
            tile.show()

    def move_tiles(self) -> None:
        """
        Moves the tiles in the direction specified by `move_direction` and handles merging of tiles.
        """
        self.tile_moved = False
        for tile in self.tiles:
            tile.already_increased = False

        if np.any(self.move_direction != 0):
            sorting_order = self.calculate_sorting_order()
            for order in sorting_order:
                for tile in self.tiles:
                    if np.array_equal(tile.position, order):
                        self.process_tile_movement(tile)
            if self.tile_moved:
                logging.info("Tiles moved")
            else:
                logging.info("No tiles moved")
        else:
            logging.debug("Move direction is zero; no tiles moved")

    def calculate_sorting_order(self) -> List[np.array]:
        sorting_vec = (
            np.array([3, 0])
            if self.move_direction[0] == 1
            else (
                np.array([0, 0])
                if self.move_direction[0] == -1
                else (
                    np.array([0, 3])
                    if self.move_direction[1] == 1
                    else np.array([0, 0])
                )
            )
        )
        vert = self.move_direction[1] != 0
        sorting_order = []
        for i in range(4):
            for j in range(4):
                temp = sorting_vec.copy()
                if vert:
                    temp[0] += j
                else:
                    temp[1] += j
                sorting_order.append(temp)
            sorting_vec -= self.move_direction
        return sorting_order

    def process_tile_movement(self, tile: Tile) -> None:
        move_to = tile.position + self.move_direction
        while self.is_position_empty(move_to):
            tile.move_to(move_to)
            move_to += self.move_direction
            self.tile_moved = True
        self.handle_potential_merge(tile, move_to)

    def is_position_empty(self, position: np.array) -> bool:
        return all(not np.array_equal(t.position, position) for t in self.tiles)

    def handle_potential_merge(self, tile: Tile, position: np.array) -> None:
        other = self.get_tile_at(position)
        if other and other.value == tile.value and not other.already_increased:
            tile.move_to(position)
            tile.death_on_impact = True
            other.already_increased = True
            other.value *= 2
            self.score += other.value
            other.set_color()
            self.tile_moved = True

    def get_tile_at(self, position: np.array) -> Optional[Tile]:
        for tile in self.tiles:
            if np.array_equal(tile.position, position):
                return tile
        return None

    def get_value(self, x: int, y: int) -> int:
        tile = self.get_tile_at(np.array([x, y]))
        return tile.value if tile else 0

    def move(self) -> None:
        if self.moving_tiles:
            for tile in self.tiles:
                tile.position += self.move_direction
            if self.done_moving():
                self.tiles = [tile for tile in self.tiles if not tile.death_on_impact]
                self.moving_tiles = False
                self.set_empty_positions()
                self.add_new_tile_not_random()

    def done_moving(self) -> bool:
        return all(not tile.death_on_impact for tile in self.tiles)

    def update(self) -> None:
        self.move()

    def set_tiles_from_history(self) -> None:
        self.tiles.clear()
        for i in range(2):
            pos = self.starting_positions[i, :2].astype(int)
            val = self.starting_positions[i, 2]
            tile = Tile(pos[0], pos[1], val)
            self.tiles.append(tile)
        self.remove_occupied_from_empty_positions()

    def remove_occupied_from_empty_positions(self) -> None:
        occupied_positions = [tile.position for tile in self.tiles]
        self.empty_positions = [
            pos
            for pos in self.empty_positions
            if not any(np.array_equal(pos, o_pos) for o_pos in occupied_positions)
        ]
