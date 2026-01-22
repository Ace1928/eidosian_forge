# Import required modules
from typing import (
    List,
    Optional,
    Tuple,
    Set,
    Dict,
)  # Import specific types from typing module
import pygame as pg  # Import pygame module and alias it as pg
from pygame.math import Vector2  # Import Vector2 class from pygame.math module
from random import randint  # Import randint function from random module
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality

# Initialize Pygame
pg.init()
# Initialize the display
pg.display.init()
# Retrieve the current display information
display_info = pg.display.Info()


# Calculate the block size based on screen resolution to ensure visibility and proportionality
def calculate_block_size(screen_width: int, screen_height: int) -> int:
    """
    Calculate the block size based on the screen resolution.

    Args:
        screen_width: The width of the screen in pixels.
        screen_height: The height of the screen in pixels.

    Returns:
        The calculated block size as an integer.
    """
    # Define the reference resolution and corresponding block size
    reference_resolution = (1920, 1080)
    reference_block_size = 20

    # Calculate the scaling factor based on the reference resolution
    scaling_factor_width = screen_width / reference_resolution[0]
    scaling_factor_height = screen_height / reference_resolution[1]
    scaling_factor = min(scaling_factor_width, scaling_factor_height)

    # Calculate the block size dynamically based on the screen size
    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))

    # Ensure the block size does not become too large or too small
    # Set minimum block size to 1x1 pixels and maximum to 30x30 pixels
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)
    return adjusted_block_size


# Apply the calculated block size based on the current screen resolution
BLOCK_SIZE = calculate_block_size(display_info.current_w, display_info.current_h)

# Define the border width as equivalent to 3 blocks
BORDER_WIDTH = 3 * BLOCK_SIZE  # Width of the border to be subtracted from each side

# Define the screen size with a proportional border around the edges
SCREEN_SIZE = (
    display_info.current_w - 2 * BORDER_WIDTH,
    display_info.current_h - 2 * BORDER_WIDTH,
)

# Define a constant for the border color as solid white
BORDER_COLOR = (255, 255, 255)  # RGB color code for white

# Instantiate the Snake object
SNAKE = snake.Snake()

# Instantiate the Apple object
APPLE = apple.Apple()

# Instantiate the Search object with snake and apple as parameters
SEARCH = search.Search(SNAKE, APPLE)

# Instantiate the Clock object for controlling the game's frame rate
CLOCK = pg.time.Clock()

# Define the desired frames per second
FPS = 60

# Calculate the tick rate based on the desired FPS
TICK_RATE = 1000 // FPS


def setup() -> Tuple[pg.Surface, "Pathfinder", pg.time.Clock]:
    """
    Initialize the game environment, setting up the display and instantiating game objects.

    Returns:
        A tuple containing the screen surface, snake object, apple object, search object, and clock object.
    """
    # Initialize Pygame
    pg.init()
    # Set the screen size using the SCREEN_SIZE constant defined globally
    screen: pg.Surface = pg.display.set_mode(SCREEN_SIZE)
    # Instantiate the Snake object using the globally defined SNAKE
    snake_object = SNAKE
    # Instantiate the Apple object using the globally defined APPLE
    apple_object = APPLE
    # Instantiate the Search object with snake and apple as parameters using the globally defined SEARCH
    search_object = Pathfinder(SCREEN_SIZE[0], SCREEN_SIZE[1], logging.getLogger())
    # Initiate the pathfinding algorithm
    search_object.get_path()
    # Utilize the globally defined CLOCK for controlling the game's frame rate
    clock_object: pg.time.Clock = CLOCK
    return screen, snake_object, apple_object, search_object, clock_object


class Pathfinder:
    """
    A class for pathfinding in a grid-based environment.
    """

    def __init__(self, width: int, height: int, logger: logging.Logger):
        """
        Initialize the Pathfinder object.

        Args:
            width: The width of the grid.
            height: The height of the grid.
            logger: The logger object for logging messages.
        """
        self.width = width
        self.height = height
        self.logger = logger
        self.obstacles: Set[Tuple[int, int]] = set()

    def calculate_distance(self, position1: Vector2, position2: Vector2) -> float:
        """
        Calculate the Manhattan distance between two positions.

        Args:
            position1: The first position as a Vector2 object.
            position2: The second position as a Vector2 object.

        Returns:
            The Manhattan distance between the two positions.
        """
        return (position1 - position2).length_manhattan()

    def calculate_obstacle_proximity(
        self,
        position: Vector2,
        obstacles: Set[Vector2],
        space_around_obstacles: int,
    ) -> float:
        """
        Calculate a penalty based on the proximity to obstacles.

        Args:
            position: The current position as a Vector2 object.
            obstacles: A set of obstacle positions as Vector2 objects.
            space_around_obstacles: The desired space to maintain around obstacles.

        Returns:
            The calculated penalty based on proximity to obstacles.
        """
        penalty = 0.0
        for obstacle in obstacles:
            distance = self.calculate_distance(position, obstacle)
            if distance <= space_around_obstacles:
                penalty += 1 / (distance + 1)
        return penalty

    def calculate_boundary_proximity(
        self,
        position: Vector2,
        boundaries: Tuple[int, int, int, int],
        space_around_boundaries: int,
    ) -> float:
        """
        Calculate a penalty based on the proximity to boundaries.

        Args:
            position: The current position as a Vector2 object.
            boundaries: The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).
            space_around_boundaries: The desired space to maintain around boundaries.

        Returns:
            The calculated penalty based on proximity to boundaries.
        """
        x_min, y_min, x_max, y_max = boundaries
        min_dist_to_boundary = min(
            position.x - x_min,
            x_max - position.x,
            position.y - y_min,
            y_max - position.y,
        )
        if min_dist_to_boundary < space_around_boundaries:
            return (space_around_boundaries - min_dist_to_boundary) ** 2
        return 0.0

    def calculate_body_position_proximity(
        self,
        position: Vector2,
        body_positions: Set[Vector2],
        space_around_agent: int,
    ) -> float:
        """
        Calculate a penalty for being too close to the snake's own body.

        Args:
            position: The current position as a Vector2 object.
            body_positions: A set of positions occupied by the snake's body as Vector2 objects.
            space_around_agent: The desired space to maintain around the snake's body.

        Returns:
            The calculated penalty for being too close to the snake's body.
        """
        penalty = 0.0
        for body_pos in body_positions:
            if self.calculate_distance(position, body_pos) < space_around_agent:
                penalty += float("inf")
        return penalty

    def evaluate_escape_routes(
        self,
        position: Vector2,
        obstacles: Set[Vector2],
        boundaries: Tuple[int, int, int, int],
    ) -> float:
        """
        Evaluate and score the availability of escape routes.

        Args:
            position: The current position as a Vector2 object.
            obstacles: A set of obstacle positions as Vector2 objects.
            boundaries: The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).

        Returns:
            The score based on the availability of escape routes.
        """
        score = 0.0
        directions = [Vector2(0, 1), Vector2(1, 0), Vector2(0, -1), Vector2(-1, 0)]
        for direction in directions:
            neighbor = position + direction
            if neighbor not in obstacles and self.is_within_boundaries(
                neighbor, boundaries
            ):
                score += 1.0
        return -score

    def is_within_boundaries(
        self, position: Vector2, boundaries: Tuple[int, int, int, int]
    ) -> bool:
        """
        Check if a position is within the specified boundaries.

        Args:
            position: The position to check as a Vector2 object.
            boundaries: The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max).

        Returns:
            True if the position is within the boundaries, False otherwise.
        """
        x_min, y_min, x_max, y_max = boundaries
        return x_min <= position.x <= x_max and y_min <= position.y <= y_max

    def apply_zigzagging_effect(self, current_heuristic: float) -> float:
        """
        Modify the heuristic to account for zigzagging, making the path less predictable.

        Args:
            current_heuristic: The current heuristic value.

        Returns:
            The modified heuristic value accounting for zigzagging.
        """
        return current_heuristic * 1.05

    def apply_dense_packing_effect(self, current_heuristic: float) -> float:
        """
        Modify the heuristic to handle dense packing scenarios more effectively.

        Args:
            current_heuristic: The current heuristic value.

        Returns:
            The modified heuristic value accounting for dense packing.
        """
        return current_heuristic * 0.95

    def heuristic(
        self,
        self_position: Vector2,
        goal_position: Vector2,
        secondary_goal_position: Optional[Vector2] = None,
        tertiary_goal_position: Optional[Vector2] = None,
        quaternary_goal_position: Optional[Vector2] = None,
        environment_boundaries: Tuple[int, int, int, int] = (0, 0, 10, 10),
        space_around_agent: int = 0,
        space_around_goals: int = 0,
        space_around_obstacles: int = 0,
        space_around_boundaries: int = 0,
        obstacles: Set[Vector2] = set(),
        escape_route_availability: bool = False,
        enhancements: List[str] = ["zigzagging"],
        dense_packing: bool = True,
        body_size_adaptations: bool = True,
        self_body_positions: Set[Vector2] = set(),
    ) -> float:
        """
        Calculate the heuristic value for the Dynamic Pathfinding algorithm using a dynamic, adaptive, multifaceted approach.

        This heuristic is optimized for real-time performance and scalability, incorporating multiple factors such as
        directional bias, obstacle avoidance, boundary awareness, body avoidance, escape route availability, dense packing,
        and path-specific adjustments. The heuristic is designed to generate strategic, efficient paths that adapt to the
        current surrounding/grid/environment/game state and adjust accordingly and efficiently.

        Args:
            self_position: The current position of the agent in the grid.
            goal_position: The target position the agent aims to reach.
            secondary_goal_position: An optional secondary target position.
            tertiary_goal_position: An optional tertiary target position.
            quaternary_goal_position: An optional quaternary target position.
            environment_boundaries: The boundaries of the grid/environment/game state.
            space_around_agent: The space around the agent to consider for path planning.
            space_around_goals: The space around the goals to consider for path planning.
            space_around_obstacles: The space around the obstacles to consider for path planning.
            space_around_boundaries: The space around the boundaries to consider for path planning.
            obstacles: The positions of obstacles in the grid.
            escape_route_availability: Whether escape routes should be considered.
            enhancements: The enhancements to apply to the path.
            dense_packing: Whether dense packing scenarios should be considered.
            body_size_adaptations: Whether the agent's body size should be considered.
            self_body_positions: The positions occupied by the agent's body (if any).

        Returns:
            The calculated heuristic value for the current state.
        """
        heuristic_value = 0.0

        # Calculate the distance to the primary goal and any secondary goals
        heuristic_value += self.calculate_distance(self_position, goal_position)
        if secondary_goal_position:
            heuristic_value += 0.5 * self.calculate_distance(
                self_position, secondary_goal_position
            )
        if tertiary_goal_position:
            heuristic_value += 0.3 * self.calculate_distance(
                self_position, tertiary_goal_position
            )
        if quaternary_goal_position:
            heuristic_value += 0.1 * self.calculate_distance(
                self_position, quaternary_goal_position
            )

        # Adjust heuristic based on the proximity to obstacles and boundaries
        heuristic_value += self.calculate_obstacle_proximity(
            self_position, obstacles, space_around_obstacles
        )
        heuristic_value += self.calculate_boundary_proximity(
            self_position, environment_boundaries, space_around_boundaries
        )

        # Consider agent's body positions if body size adaptations are enabled
        if body_size_adaptations:
            heuristic_value += self.calculate_body_position_proximity(
                self_position, self_body_positions, space_around_agent
            )

        # Factor in escape routes availability
        if escape_route_availability:
            heuristic_value += self.evaluate_escape_routes(
                self_position, obstacles, environment_boundaries
            )

        # Apply enhancements to the heuristic calculation
        for enhancement in enhancements:
            if enhancement == "zigzagging":
                heuristic_value = self.apply_zigzagging_effect(heuristic_value)
            elif enhancement == "dense_packing":
                heuristic_value = self.apply_dense_packing_effect(heuristic_value)

        # Log the calculated heuristic value
        self.logger.debug(f"Calculated heuristic value: {heuristic_value}")

        return heuristic_value

    def a_star_search(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Implement the A* algorithm to find the optimal path from start to goal.

        Args:
            start: The starting position as a tuple of (x, y) coordinates.
            goal: The goal position as a tuple of (x, y) coordinates.

        Returns:
            The optimal path from start to goal as a list of (x, y) coordinates.
        """
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, start, goal)

            for next_pos in self.neighbors(current):
                new_cost = current_cost + self.heuristic(next_pos, goal)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal)
                    heapq.heappush(open_set, (priority, new_cost, next_pos))
                    came_from[next_pos] = current

        return []

    def reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to goal using the came_from map.

        Args:
            came_from: A dictionary mapping each position to its previous position in the path.
            start: The starting position as a tuple of (x, y) coordinates.
            goal: The goal position as a tuple of (x, y) coordinates.

        Returns:
            The reconstructed path from start to goal as a list of (x, y) coordinates.
        """
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generate the neighbors of a node considering boundaries and obstacles.

        Args:
            node: The current node as a tuple of (x, y) coordinates.

        Returns:
            A list of neighboring positions as tuples of (x, y) coordinates.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        result = []
        for dx, dy in directions:
            nx, ny = node[0] + dx, node[1] + dy
            if (
                0 <= nx < self.width
                and 0 <= ny < self.height
                and (nx, ny) not in self.obstacles
            ):
                result.append((nx, ny))
        return result


def get_neighbors(
    node: Tuple[int, int], width: int, height: int, obstacles: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Generate the neighbors of a node considering boundaries and obstacles.

    Args:
        node: The current node as a tuple of (x, y) coordinates.
        width: The width of the grid.
        height: The height of the grid.
        obstacles: A set of obstacle positions as tuples of (x, y) coordinates.

    Returns:
        A list of neighboring positions as tuples of (x, y) coordinates.
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    result = []
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
            result.append((nx, ny))
    return result
