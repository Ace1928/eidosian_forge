# Import required modules
from typing import (
    List,
    Optional,
    Tuple,
    Set,
    Dict,
)  # Import specific types from typing module for type hinting
import pygame  # Import pygame module for game development
from pygame.math import (
    Vector2,
)  # Import Vector2 class from pygame.math module for vector calculations
from random import (
    randint,
)  # Import randint function from random module for generating random numbers
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality

# Initialize Pygame and its display module
pygame.init()
pygame.display.init()

# Retrieve the current display information
display_info = pygame.display.Info()


def calculate_block_size(screen_width: int, screen_height: int) -> int:
    """
    Calculate the block size based on the screen resolution.

    This function calculates the block size dynamically based on the screen resolution
    to ensure visibility and proportionality. It takes the screen width and height as
    input and returns the calculated block size as an integer.

    Args:
        screen_width (int): The width of the screen in pixels.
        screen_height (int): The height of the screen in pixels.

    Returns:
        int: The calculated block size.
    """
    # Define the reference resolution and corresponding block size
    reference_resolution = Vector2(1920, 1080)
    reference_block_size = 20

    # Calculate the scaling factor based on the reference resolution
    scaling_factor = min(
        screen_width / reference_resolution.x, screen_height / reference_resolution.y
    )

    # Calculate the block size dynamically based on the screen size
    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))

    # Ensure the block size does not become too large or too small
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

# Instantiate the Clock object for controlling the game's frame rate
CLOCK = pygame.time.Clock()

# Define the desired frames per second
FRAMES_PER_SECOND = 60

# Calculate the tick rate based on the desired FPS
TICK_RATE = 1000 // FRAMES_PER_SECOND


def setup() -> Tuple[pygame.Surface, "Pathfinder", pygame.time.Clock]:
    """
    Initialize the game environment, setting up the display and instantiating game objects.

    This function initializes Pygame, sets up the screen surface, instantiates the Pathfinder
    object, initiates the pathfinding algorithm, and returns the necessary objects for the game.

    Returns:
        Tuple[pygame.Surface, Pathfinder, pygame.time.Clock]:
            - screen (pygame.Surface): The game screen surface.
            - search_object (Pathfinder): The Pathfinder object for pathfinding.
            - clock_object (pygame.time.Clock): The clock object for controlling the game's frame rate.
    """
    # Initialize Pygame
    pygame.init()
    # Set the screen size using the SCREEN_SIZE constant defined globally
    screen: pygame.Surface = pygame.display.set_mode(SCREEN_SIZE)
    # Instantiate the Pathfinder object with screen size and logger
    search_object = Pathfinder(SCREEN_SIZE[0], SCREEN_SIZE[1], logging.getLogger())
    # Initiate the pathfinding algorithm
    search_object.find_path()
    # Utilize the globally defined CLOCK for controlling the game's frame rate
    clock_object: pygame.time.Clock = CLOCK
    return screen, search_object, clock_object


class Pathfinder:
    """
    A class dedicated to pathfinding within a grid-based environment using the A* algorithm.

    This class handles the computation of paths considering various environmental factors such as obstacles,
    boundaries, and the snake's own body positions. It uses the Euclidean distance for calculations and
    incorporates penalties for proximity to obstacles and boundaries.

    Attributes:
        grid_width (int): The width of the grid in which pathfinding is performed. Default is 100.
        grid_height (int): The height of the grid in which pathfinding is performed. Default is 100.
        logger (logging.Logger): Logger for recording operational logs.
        obstacles (Set[Vector2]): A set of obstacles within the grid, represented by their positions.
    """

    def __init__(
        self,
        grid_width: int = 100,
        grid_height: int = 100,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the Pathfinder object with the necessary dimensions and logger.

        Args:
            grid_width (int): The width of the grid. Default is 100.
            grid_height (int): The height of the grid. Default is 100.
            logger (logging.Logger): The logger object for logging messages. Default is None.
        """
        # Set the width of the grid
        self.grid_width: int = grid_width
        # Set the height of the grid
        self.grid_height: int = grid_height
        # Set the logger, using a default logger if none is provided
        self.logger: logging.Logger = (
            logger if logger is not None else logging.getLogger(__name__)
        )

    def calculate_euclidean_distance(
        self, position1: Vector2, position2: Vector2
    ) -> float:
        """
        Calculate the Euclidean distance between two positions in the grid.

        This function utilizes Pygame's Vector2 object to calculate the Euclidean distance between two positions.
        The Euclidean distance is more suitable for grid-based pathfinding compared to Manhattan distance in certain scenarios.

        Args:
            position1 (Vector2): The first position vector.
            position2 (Vector2): The second position vector.

        Returns:
            float: The Euclidean distance between the two positions.
        """
        # Use Pygame's built-in distance_to method for efficient distance calculation
        return position1.distance_to(position2)

    def calculate_obstacle_proximity_penalty(
        self, position: Vector2, space_around_obstacles: int = 5
    ) -> float:
        """
        Calculate a penalty score based on the proximity to the nearest obstacle.

        This function iterates through each obstacle within the line of sight and calculates the distance to the given position.
        If the distance is less than the specified space around obstacles, a penalty is calculated based on the inverse of the distance.
        Closer obstacles are assigned a higher penalty to emphasize their significance.

        Args:
            position (Vector2): The current position as a Vector2 object.
            space_around_obstacles (int): The minimum desired distance from any obstacle. Default is 5.

        Returns:
            float: The total penalty accumulated from all nearby obstacles within the line of sight.
        """
        # Initialize the penalty to 0
        penalty: float = 0.0
        # Get the obstacles within the line of sight
        visible_obstacles: Set[Vector2] = self.get_line_of_sight_obstacles(position)
        # Iterate through each visible obstacle
        for obstacle in visible_obstacles:
            # Calculate the distance between the current position and the obstacle
            distance: float = self.calculate_euclidean_distance(position, obstacle)
            # Check if the distance is within the specified space around obstacles
            if distance <= space_around_obstacles:
                # Calculate the penalty based on the inverse of the distance (closer obstacles have higher penalty)
                # Adding 1 to the distance to avoid division by zero
                penalty += 1 / (distance + 1)
        # Return the total penalty accumulated from all nearby obstacles within the line of sight
        return penalty

    def calculate_boundary_proximity_penalty(
        self,
        position: Vector2,
        boundaries: Tuple[int, int, int, int] = (0, 0, 100, 100),
        space_around_boundaries: int = 5,
    ) -> float:
        """
        Calculate a penalty based on the proximity to boundaries.

        This function computes a penalty score based on how close the given position is to the boundaries of the environment.
        The penalty increases as the position approaches the boundary within a specified threshold.

        Args:
            position (Vector2): The current position as a Vector2 object.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max). Default is (0, 0, 100, 100).
            space_around_boundaries (int): The desired space to maintain around boundaries. Default is 5.

        Returns:
            float: The calculated penalty based on proximity to boundaries.
        """
        # Unpack the boundary values from the tuple
        x_min, y_min, x_max, y_max = boundaries
        # Calculate the minimum distance to any boundary
        min_distance_to_boundary: float = min(
            position.x - x_min,
            x_max - position.x,
            position.y - y_min,
            y_max - position.y,
        )
        # Check if the position is within the specified space around boundaries
        if min_distance_to_boundary < space_around_boundaries:
            # Calculate the penalty based on the square of the difference between the desired space and the actual distance
            return (space_around_boundaries - min_distance_to_boundary) ** 2
        # Return 0 penalty if the position is not within the specified space around boundaries
        return 0.0

    def calculate_body_position_proximity_penalty(
        self,
        position: Vector2,
        body_positions: Set[Vector2],
        space_around_agent: int = 2,
    ) -> float:
        """
        Calculate a penalty for being too close to the snake's own body.

        This function iterates through each position occupied by the snake's body within the line of sight
        and calculates a penalty if the given position is within a specified distance from any part of the body.
        The penalty is set to infinity to represent an impassable barrier.

        Args:
            position (Vector2): The current position as a Vector2 object.
            body_positions (Set[Vector2]): The positions occupied by the snake's body.
            space_around_agent (int): The desired space to maintain around the snake's body. Default is 2.

        Returns:
            float: The calculated penalty for being too close to the snake's body.
        """
        # Initialize the penalty to 0
        penalty: float = 0.0
        # Get the body positions within the line of sight
        visible_body_positions: Set[Vector2] = self.get_line_of_sight_body_positions(
            position
        )
        # Iterate through each visible body position
        for body_position in visible_body_positions:
            # Check if the distance between the current position and the body position is less than the specified space around the agent
            if (
                self.calculate_euclidean_distance(position, body_position)
                < space_around_agent
            ):
                # Set the penalty to infinity to represent an impassable barrier
                penalty += float("inf")
        # Return the calculated penalty for being too close to the snake's body
        return penalty

    def evaluate_escape_routes(
        self,
        position: Vector2,
        obstacles: Set[Vector2],
        boundaries: Tuple[int, int, int, int] = (0, 0, 100, 100),
    ) -> float:
        """
        Evaluate and score the availability of escape routes.

        This function assesses the number of available escape routes from the current position.
        It checks each cardinal direction (up, down, left, right) and scores based on the number of unobstructed paths.

        Args:
            position (Vector2): The current position as a Vector2 object.
            obstacles (Set[Vector2]): The positions of obstacles in the environment.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max). Default is (0, 0, 100, 100).

        Returns:
            float: The score based on the availability of escape routes.
        """
        # Initialize the score to 0
        score: float = 0.0
        # Define the cardinal directions as Vector2 objects
        directions: List[Vector2] = [
            Vector2(0, 1),  # Up
            Vector2(1, 0),  # Right
            Vector2(0, -1),  # Down
            Vector2(-1, 0),  # Left
        ]
        # Iterate through each direction
        for direction in directions:
            # Calculate the neighboring position by adding the direction to the current position
            neighbor: Vector2 = position + direction
            # Check if the neighboring position is not an obstacle and is within the boundaries
            if neighbor not in obstacles and self.is_position_within_boundaries(
                neighbor, boundaries
            ):
                # Increment the score by 1 for each available escape route
                score += 1.0
        # Return the negative score to represent fewer escape routes (higher score means fewer escape routes)
        return -score

    def is_position_within_boundaries(
        self,
        position: Vector2,
        boundaries: Tuple[int, int, int, int] = (0, 0, 100, 100),
    ) -> bool:
        """
        Check if a position is within the specified boundaries.

        This function determines whether a given position falls within the defined environmental boundaries.

        Args:
            position (Vector2): The position to check as a Vector2 object.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max). Default is (0, 0, 100, 100).

        Returns:
            bool: True if the position is within the boundaries, False otherwise.
        """
        # Unpack the boundary values from the tuple
        x_min, y_min, x_max, y_max = boundaries
        # Check if the position's x-coordinate is within the x-boundaries and the y-coordinate is within the y-boundaries
        return x_min <= position.x <= x_max and y_min <= position.y <= y_max

    def apply_zigzagging_effect(self, current_heuristic: float = 1.0) -> float:
        """
        Modify the heuristic to account for zigzagging, making the path less predictable.

        This function increases the heuristic value slightly to account for the added complexity of zigzagging,
        which can make the path less predictable and potentially safer from pursuers.

        Args:
            current_heuristic (float): The current heuristic value. Default is 1.0.

        Returns:
            float: The modified heuristic value accounting for zigzagging.
        """
        # Increase the current heuristic value by 5% to account for zigzagging
        return current_heuristic * 1.05

    def apply_dense_packing_effect(self, current_heuristic: float = 1.0) -> float:
        """
        Modify the heuristic to handle dense packing scenarios more effectively.

        This function decreases the heuristic value to account for dense packing scenarios,
        where closer packing might be necessary or unavoidable.

        Args:
            current_heuristic (float): The current heuristic value. Default is 1.0.

        Returns:
            float: The modified heuristic value accounting for dense packing.
        """
        # Decrease the current heuristic value by 5% to account for dense packing
        return current_heuristic * 0.95

    def get_line_of_sight_obstacles(
        self, position: Vector2, sight_range: int = 5
    ) -> Set[Vector2]:
        """
        Dynamically calculate obstacles within the line of sight of the agent.

        This function determines the obstacles that are visible to the agent based on its current position and sight range.
        It checks the surrounding positions within the sight range and adds any obstacles found to the set of visible obstacles.

        Args:
            position (pygame.math.Vector2): The current position of the agent as a Vector2 object.
            sight_range (int): The range of sight for the agent. Default is 5.

        Returns:
            Set[pygame.math.Vector2]: A set of obstacle positions within the line of sight of the agent.
        """
        # Initialize an empty set to store the visible obstacles
        visible_obstacles: Set[pygame.math.Vector2] = set()
        # Iterate through the positions within the sight range
        for x in range(
            max(0, int(position.x) - sight_range),
            min(self.grid_width, int(position.x) + sight_range + 1),
        ):
            for y in range(
                max(0, int(position.y) - sight_range),
                min(self.grid_height, int(position.y) + sight_range + 1),
            ):
                # Create a Vector2 object for the current position
                current_position: pygame.math.Vector2 = pygame.math.Vector2(x, y)
                # Check if the current position is an obstacle and add it to the set of visible obstacles
                if self.is_position_obstacle(current_position):
                    visible_obstacles.add(current_position)
        # Return the set of visible obstacles
        return visible_obstacles

    def get_line_of_sight_body_positions(
        self, position: pygame.math.Vector2, sight_range: int = 5
    ) -> Set[pygame.math.Vector2]:
        """
        Dynamically calculate the snake's body positions within the line of sight of the agent.

        This function determines the positions occupied by the snake's body that are visible to the agent based on its current position and sight range.
        It checks the surrounding positions within the sight range and adds any body positions found to the set of visible body positions.

        Args:
            position (pygame.math.Vector2): The current position of the agent as a Vector2 object.
            sight_range (int): The range of sight for the agent. Default is 5.

        Returns:
            Set[pygame.math.Vector2]: A set of body positions within the line of sight of the agent.
        """
        # Initialize an empty set to store the visible body positions
        visible_body_positions: Set[pygame.math.Vector2] = set()
        # Iterate through the positions within the sight range
        for x in range(
            max(0, int(position.x) - sight_range),
            min(self.grid_width, int(position.x) + sight_range + 1),
        ):
            for y in range(
                max(0, int(position.y) - sight_range),
                min(self.grid_height, int(position.y) + sight_range + 1),
            ):
                # Create a Vector2 object for the current position
                current_position: pygame.math.Vector2 = pygame.math.Vector2(x, y)
                # Check if the current position is a body position and add it to the set of visible body positions
                if self.is_position_body(current_position):
                    visible_body_positions.add(current_position)
        # Return the set of visible body positions
        return visible_body_positions

    def get_line_of_sight_goals(
        self, position: pygame.math.Vector2, sight_range: int = 5
    ) -> Set[pygame.math.Vector2]:
        """
        Dynamically calculate the goal positions within the line of sight of the agent.

        This function determines the goal positions that are visible to the agent based on its current position and sight range.
        It checks the surrounding positions within the sight range and adds any goal positions found to the set of visible goals.

        Args:
            position (pygame.math.Vector2): The current position of the agent as a Vector2 object.
            sight_range (int): The range of sight for the agent. Default is 5.

        Returns:
            Set[pygame.math.Vector2]: A set of goal positions within the line of sight of the agent.
        """
        # Initialize an empty set to store the visible goal positions
        visible_goal_positions: Set[pygame.math.Vector2] = set()
        # Iterate through the positions within the sight range
        for x in range(
            max(0, int(position.x) - sight_range),
            min(self.grid_width, int(position.x) + sight_range + 1),
        ):
            for y in range(
                max(0, int(position.y) - sight_range),
                min(self.grid_height, int(position.y) + sight_range + 1),
            ):
                # Create a Vector2 object for the current position
                current_position: pygame.math.Vector2 = pygame.math.Vector2(x, y)
                # Check if the current position is a goal position and add it to the set of visible goal positions
                if current_position in self.goal_positions:
                    visible_goal_positions.add(current_position)
        # Return the set of visible goal positions
        return visible_goal_positions

    def heuristic(
        self,
        current_position: pygame.math.Vector2,
        goal_position: pygame.math.Vector2,
        secondary_goal_position: pygame.math.Vector2 = None,
        tertiary_goal_position: pygame.math.Vector2 = None,
        quaternary_goal_position: pygame.math.Vector2 = None,
        environment_boundaries: Tuple[int, int, int, int] = (0, 0, 100, 100),
        space_around_agent: int = 0,
        space_around_goals: int = 0,
        space_around_obstacles: int = 0,
        space_around_boundaries: int = 0,
        obstacles: Set[pygame.math.Vector2] = set(),
        escape_route_availability: bool = False,
        enhancements: List[str] = None,
        dense_packing: bool = True,
        body_size_adaptations: bool = True,
        self_body_positions: Set[pygame.math.Vector2] = set(),
    ) -> float:
        """
        Calculate the heuristic value for the Dynamic Pathfinding algorithm.

        This heuristic function incorporates multiple factors to determine the estimated cost
        from the current position to the goal position. It takes into account the primary goal
        position, as well as optional secondary, tertiary, and quaternary goal positions.
        The heuristic value is adjusted based on the proximity to obstacles, boundaries, and
        the agent's own body positions. It also considers factors such as escape route availability,
        dense packing scenarios, and body size adaptations.

        Args:
            current_position (pygame.math.Vector2): The current position of the agent.
            goal_position (pygame.math.Vector2): The primary target position the agent aims to reach.
            secondary_goal_position (pygame.math.Vector2, optional): Secondary target position. Defaults to None.
            tertiary_goal_position (pygame.math.Vector2, optional): Tertiary target position. Defaults to None.
            quaternary_goal_position (pygame.math.Vector2, optional): Quaternary target position. Defaults to None.
            environment_boundaries (Tuple[int, int, int, int], optional): The boundaries of the environment.
                Defaults to (0, 0, 100, 100).
            space_around_agent (int, optional): The desired space to maintain around the agent. Defaults to 0.
            space_around_goals (int, optional): The desired space to maintain around goal positions. Defaults to 0.
            space_around_obstacles (int, optional): The desired space to maintain around obstacles. Defaults to 0.
            space_around_boundaries (int, optional): The desired space to maintain around boundaries. Defaults to 0.
            obstacles (Set[pygame.math.Vector2], optional): The positions of obstacles in the environment. Defaults to set().
            escape_route_availability (bool, optional): Flag indicating whether escape routes should be considered. Defaults to False.
            enhancements (List[str], optional): List of enhancements to apply to the heuristic calculation. Defaults to None.
            dense_packing (bool, optional): Flag indicating whether dense packing scenarios should be considered. Defaults to True.
            body_size_adaptations (bool, optional): Flag indicating whether body size adaptations should be considered. Defaults to True.
            self_body_positions (Set[pygame.math.Vector2], optional): The positions occupied by the agent's own body. Defaults to set().

        Returns:
            float: The calculated heuristic value estimating the cost from the current position to the goal position.
        """
        # Initialize the heuristic value to 0.0
        heuristic_value: float = 0.0

        # Calculate the Euclidean distance from the current position to the primary goal position
        primary_goal_distance: float = self.calculate_euclidean_distance(
            current_position, goal_position
        )
        # Add the primary goal distance to the heuristic value
        heuristic_value += primary_goal_distance

        # If a secondary goal position is provided
        if secondary_goal_position is not None:
            # Calculate the Euclidean distance from the current position to the secondary goal position
            secondary_goal_distance: float = self.calculate_euclidean_distance(
                current_position, secondary_goal_position
            )
            # Add the weighted secondary goal distance to the heuristic value
            heuristic_value += 0.5 * secondary_goal_distance

        # If a tertiary goal position is provided
        if tertiary_goal_position is not None:
            # Calculate the Euclidean distance from the current position to the tertiary goal position
            tertiary_goal_distance: float = self.calculate_euclidean_distance(
                current_position, tertiary_goal_position
            )
            # Add the weighted tertiary goal distance to the heuristic value
            heuristic_value += 0.3 * tertiary_goal_distance

        # If a quaternary goal position is provided
        if quaternary_goal_position is not None:
            # Calculate the Euclidean distance from the current position to the quaternary goal position
            quaternary_goal_distance: float = self.calculate_euclidean_distance(
                current_position, quaternary_goal_position
            )
            # Add the weighted quaternary goal distance to the heuristic value
            heuristic_value += 0.1 * quaternary_goal_distance

        # Calculate the obstacle proximity penalty using the calculate_obstacle_proximity_penalty function
        obstacle_proximity_penalty: float = self.calculate_obstacle_proximity_penalty(
            current_position, space_around_obstacles
        )
        # Add the obstacle proximity penalty to the heuristic value
        heuristic_value += obstacle_proximity_penalty

        # Calculate the boundary proximity penalty using the calculate_boundary_proximity_penalty function
        boundary_proximity_penalty: float = self.calculate_boundary_proximity_penalty(
            current_position, environment_boundaries, space_around_boundaries
        )
        # Add the boundary proximity penalty to the heuristic value
        heuristic_value += boundary_proximity_penalty

        # If body size adaptations are enabled
        if body_size_adaptations:
            # Calculate the body position proximity penalty using the calculate_body_position_proximity_penalty function
            body_position_proximity_penalty: float = (
                self.calculate_body_position_proximity_penalty(
                    current_position, self_body_positions, space_around_agent
                )
            )
            # Add the body position proximity penalty to the heuristic value
            heuristic_value += body_position_proximity_penalty

        # If escape route availability is considered
        if escape_route_availability:
            # Evaluate the escape routes using the evaluate_escape_routes function
            escape_route_score: float = self.evaluate_escape_routes(
                current_position, obstacles, environment_boundaries
            )
            # Add the escape route score to the heuristic value
            heuristic_value += escape_route_score

        # If enhancements are provided
        if enhancements is not None:
            # Iterate over each enhancement
            for enhancement in enhancements:
                # If the enhancement is "zigzagging"
                if enhancement == "zigzagging":
                    # Apply the zigzagging effect to the heuristic value using the apply_zigzagging_effect function
                    heuristic_value = self.apply_zigzagging_effect(heuristic_value)
                # If the enhancement is "dense_packing"
                elif enhancement == "dense_packing":
                    # Apply the dense packing effect to the heuristic value using the apply_dense_packing_effect function
                    heuristic_value = self.apply_dense_packing_effect(heuristic_value)

        # Log the calculated heuristic value for debugging purposes
        self.logger.debug(f"Calculated heuristic value: {heuristic_value}")

        # Return the final heuristic value
        return heuristic_value

    def astar_search(
        self, start_position: pygame.math.Vector2, goal_position: pygame.math.Vector2
    ) -> List[pygame.math.Vector2]:
        """
        Implement the A* search algorithm to find the optimal path from start to goal.

        This function utilizes the A* search algorithm to find the optimal path from the start position to the goal position.
        It maintains an open set of positions to explore, and a closed set of visited positions.
        The algorithm selects the position with the lowest total cost (g_cost + h_cost) from the open set,
        explores its neighbors, and updates the costs and paths accordingly.
        It continues until the goal position is reached or there are no more positions to explore.
        The path is then extended to create a full cycle, zigzagging from the goal back to the start position.

        Args:
            start_position (pygame.math.Vector2): The starting position of the search.
            goal_position (pygame.math.Vector2): The target position to reach.

        Returns:
            List[pygame.math.Vector2]: The optimal path from start to goal and back to start as a list of positions,
                or an empty list if no path is found.
        """
        # Initialize the open set with the start position and its costs
        open_set: List[Tuple[float, float, pygame.math.Vector2]] = [
            (0, 0, start_position)
        ]

        # Initialize dictionaries to store the path and cost information
        came_from: Dict[pygame.math.Vector2, Optional[pygame.math.Vector2]] = {
            start_position: None
        }
        g_cost: Dict[pygame.math.Vector2, float] = {start_position: 0}

        # Loop until the open set is empty
        while open_set:
            # Get the position with the lowest total cost (g_cost + h_cost) from the open set
            _, current_g_cost, current_position = heapq.heappop(open_set)

            # Check if the goal position is reached
            if current_position == goal_position:
                # If the goal is reached, reconstruct the optimal path from start to goal
                path_to_goal: List[pygame.math.Vector2] = self.reconstruct_path(
                    came_from, start_position, goal_position
                )

                # Extend the path to create a full cycle, zigzagging from the goal back to the start position
                path_to_start: List[pygame.math.Vector2] = self.reconstruct_path(
                    came_from, goal_position, start_position
                )
                path_to_start.reverse()

                # Combine the path from start to goal and the path from goal to start
                full_path: List[pygame.math.Vector2] = path_to_goal + path_to_start

                # Return the full path
                return full_path

            # Explore the neighbors of the current position
            for neighbor_position in self.get_neighbors(current_position):
                # Calculate the tentative g_cost to reach the neighbor position
                tentative_g_cost: float = (
                    current_g_cost
                    + self.calculate_euclidean_distance(
                        current_position, neighbor_position
                    )
                )

                # Check if the neighbor position has not been visited or has a lower g_cost
                if (
                    neighbor_position not in g_cost
                    or tentative_g_cost < g_cost[neighbor_position]
                ):
                    # Update the g_cost and path for the neighbor position
                    g_cost[neighbor_position] = tentative_g_cost
                    h_cost: float = self.heuristic(neighbor_position, goal_position)
                    total_cost: float = tentative_g_cost + h_cost
                    heapq.heappush(
                        open_set, (total_cost, tentative_g_cost, neighbor_position)
                    )
                    came_from[neighbor_position] = current_position

        # If the open set is empty and the goal position is not reached, return an empty path
        return []

    def reconstruct_path(
        self,
        came_from: Dict[pygame.math.Vector2, Optional[pygame.math.Vector2]],
        start_position: pygame.math.Vector2,
        goal_position: pygame.math.Vector2,
    ) -> List[pygame.math.Vector2]:
        """
        Reconstruct the optimal path from start to goal using the came_from dictionary.

        This function takes the came_from dictionary, start position, and goal position,
        and reconstructs the optimal path from the start to the goal by backtracking through the came_from dictionary.
        It starts from the goal position and follows the parent positions until reaching the start position.
        The reconstructed path is then reversed to obtain the correct order from start to goal.

        Args:
            came_from (Dict[pygame.math.Vector2, Optional[pygame.math.Vector2]]): A dictionary mapping each position to its parent position in the optimal path.
            start_position (pygame.math.Vector2): The starting position of the path.
            goal_position (pygame.math.Vector2): The target position of the path.

        Returns:
            List[pygame.math.Vector2]: The reconstructed optimal path from start to goal as a list of positions.
        """
        # Initialize an empty path
        path: List[pygame.math.Vector2] = []

        # Start from the goal position
        current_position: pygame.math.Vector2 = goal_position

        # Backtrack from the goal position to the start position using the came_from dictionary
        while current_position != start_position:
            # Add the current position to the path
            path.append(current_position)
            # Move to the parent position of the current position
            current_position = came_from[current_position]

        # Add the start position to the path
        path.append(start_position)

        # Reverse the path to get the correct order from start to goal
        path.reverse()

        # Return the reconstructed path
        return path

    def get_neighbors(self, position: pygame.math.Vector2) -> List[pygame.math.Vector2]:
        """
        Get the valid neighboring positions of a given position.

        This function takes a position and returns a list of its valid neighboring positions.
        The neighboring positions are calculated by adding the four cardinal directions (up, right, down, left) to the current position.
        The validity of each neighboring position is checked against the grid boundaries and obstacle positions.
        Only the positions that are within the grid boundaries and not occupied by obstacles are considered valid neighbors.

        Args:
            position (pygame.math.Vector2): The position for which to get the neighbors.

        Returns:
            List[pygame.math.Vector2]: A list of valid neighboring positions.
        """
        # Define the four cardinal directions
        directions: List[pygame.math.Vector2] = [
            pygame.math.Vector2(0, -1),  # Up
            pygame.math.Vector2(1, 0),  # Right
            pygame.math.Vector2(0, 1),  # Down
            pygame.math.Vector2(-1, 0),  # Left
        ]

        # Initialize an empty list to store the valid neighbors
        neighbors: List[pygame.math.Vector2] = []

        # Iterate over each direction
        for direction in directions:
            # Calculate the new position by adding the direction to the current position
            new_position: pygame.math.Vector2 = position + direction

            # Check if the new position is within the grid boundaries
            if self.is_within_boundaries(new_position):
                # Check if the new position is not an obstacle
                if not self.is_obstacle(new_position):
                    # Add the new position to the list of valid neighbors
                    neighbors.append(new_position)

        # Return the list of valid neighbors
        return neighbors


class Simulation:
    def __init__(self):
        self.grid_width = 100
        self.grid_height = 100
        self.logger = logging.getLogger(__name__)
        self.pathfinder = Pathfinder(self.grid_width, self.grid_height, self.logger)
        self.screen, self.search_object, self.clock = setup()
        self.running = True
        self.start_position = Vector2(10, 10)
        self.goal_position = Vector2(90, 90)
        self.path = self.search_object.astar_search(
            self.start_position, self.goal_position
        )

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FRAMES_PER_SECOND)
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):

        self.path = self.search_object.astar_search(
            self.start_position, self.goal_position
        )

    def render(self):
        self.screen.fill((0, 0, 0))
        for position in self.path:
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (
                    position.x * BLOCK_SIZE + BORDER_WIDTH,
                    position.y * BLOCK_SIZE + BORDER_WIDTH,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                ),
            )
        pygame.display.flip()
