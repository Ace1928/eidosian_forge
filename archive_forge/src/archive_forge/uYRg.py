import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Dict, Deque, Set, Optional, Callable
import logging
import math
from heapq import heappush, heappop
from itertools import product

# Setup logging configuration
logging.basicConfig(
    filename="snake_game_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Seed the random generator for reproducible results during development
seed(0)

# Initialize pygame
pg.init()
# Set the display to fullscreen at the current display resolution
display_info = pg.display.Info()  # Get current display information
SCREEN_WIDTH: int = display_info.current_w  # Set width to current width of display
SCREEN_HEIGHT: int = display_info.current_h  # Set height to current height of display
window: pg.Surface = pg.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT), pg.FULLSCREEN
)  # Set display mode to fullscreen
logging.info(
    f"Display set to fullscreen with resolution {SCREEN_WIDTH}x{SCREEN_HEIGHT}"
)
BLOCK_SIZE: int = 20
pg.display.set_caption("Enhanced Snake Game with A* Pathfinding")
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import multiprocessing
from queue import PriorityQueue
from typing import Set, List, Tuple, Dict, Deque
from collections import deque
import math


def compute_square_difference(x: int, y: int) -> int:
    """
    Compute the square of the difference between two integers.
    This function is designed to be used in parallel processing environments to maximize efficiency.
    """
    return (x - y) ** 2


def euclidean_distance(node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points using an extremely optimized approach.
    This function maximizes the utilization of available CPU resources by employing multiprocessing,
    which allows for parallel computation of squared differences across multiple cores, ensuring
    unparalleled efficiency.

    The function uses a ProcessPoolExecutor to parallelize the computation of squared differences
    for the x and y coordinates separately. This approach is designed to optimize CPU usage and
    reduce computation time, especially beneficial for large game states and extensive search areas.
    """
    # Utilizing ProcessPoolExecutor to leverage multiple CPU cores for parallel computation
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Submitting tasks to the executor for both x and y coordinate differences
        future_x = executor.submit(compute_square_difference, node1[0], node2[0])
        future_y = executor.submit(compute_square_difference, node1[1], node2[1])

        # Retrieving results from the futures once they are completed
        result_x = future_x.result()
        result_y = future_y.result()

    # Calculating the Euclidean distance by taking the square root of the sum of squared differences
    distance = np.sqrt(result_x + result_y)
    return distance


def heuristic(
    node1: Tuple[int, int],
    node2: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    fruit_position: Tuple[int, int],
) -> float:
    """
    An advanced heuristic function that combines Euclidean distance with sophisticated game-specific adjustments.
    This heuristic is meticulously designed to be both efficient and highly effective for the snake game's pathfinding needs,
    focusing on survival, density, fruit acquisition, and space filling in a prioritized manner.

    The function employs a multi-threaded approach to compute various components of the heuristic simultaneously,
    leveraging the maximum available CPU resources to ensure optimal performance even in extensive game states and search areas.
    """
    # Utilizing ThreadPoolExecutor to maximize CPU utilization and parallelize computations
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
        # Future for base Euclidean distance calculation
        base_distance_future = executor.submit(euclidean_distance, node1, node2)

        # Futures for body proximity penalties
        body_proximity_futures = [
            executor.submit(math.exp, -euclidean_distance(node1, body_part))
            for body_part in snake_body
        ]

        # Future for calculating the center of mass of the snake body
        center_of_mass_future = executor.submit(
            lambda: tuple(np.mean(np.array(snake_body), axis=0))
        )

        # Collecting results from futures
        base_distance = base_distance_future.result()
        body_proximity_penalty = sum(f.result() for f in body_proximity_futures)
        center_of_mass = center_of_mass_future.result()

        # Additional futures dependent on the center of mass calculation
        density_bonus_future = executor.submit(
            lambda: -np.linalg.norm(np.array(node1) - np.array(center_of_mass))
        )
        fruit_priority_future = executor.submit(
            lambda: (
                -euclidean_distance(node1, fruit_position)
                if np.array_equal(node1, center_of_mass)
                else 0
            )
        )

        # Calculating zigzag penalty if the snake body has more than one segment
        if len(snake_body) > 1:
            direction_vector = np.subtract(snake_body[0], snake_body[1])
            zigzag_penalty_future = executor.submit(
                lambda: -abs(
                    np.cross(direction_vector, np.subtract(node1, snake_body[0]))
                )
            )
            zigzag_penalty = zigzag_penalty_future.result()
        else:
            zigzag_penalty = 0

        # Collecting results from additional futures
        density_bonus = density_bonus_future.result()
        fruit_priority = fruit_priority_future.result()

    # Combining all heuristic components with appropriate weights
    return (
        base_distance
        + 10 * body_proximity_penalty
        + 5 * density_bonus
        + 20 * fruit_priority
        + 15 * zigzag_penalty
    )


from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List


def is_within_boundaries(
    node: Tuple[int, int], screen_width: int, screen_height: int
) -> bool:
    """
    Check if a node is within the game boundaries.

    Args:
    node (Tuple[int, int]): The node to check.
    screen_width (int): The width of the game screen.
    screen_height (int): The height of the game screen.

    Returns:
    bool: True if the node is within the boundaries, False otherwise.
    """
    x, y = node
    return 0 <= x < screen_width and 0 <= y < screen_height


def get_neighbors(
    node: Tuple[int, int],
    block_size: int = BLOCK_SIZE,
    screen_width: int = SCREEN_WIDTH,
    screen_height: int = SCREEN_HEIGHT,
) -> List[Tuple[int, int]]:
    """
    Return the neighbors of the given node within game boundaries.
    This function considers only four possible movements (no diagonals) for strict pathfinding.

    Args:
    node (Tuple[int, int]): The node for which neighbors are to be found.
    block_size (int): The size of each movement block.
    screen_width (int): The width of the game screen.
    screen_height (int): The height of the game screen.

    Returns:
    List[Tuple[int, int]]: A list of neighbor nodes that are within the game boundaries.
    """
    directions = [
        (BLOCK_SIZE, 0),  # Right
        (-BLOCK_SIZE, 0),  # Left
        (0, BLOCK_SIZE),  # Down
        (0, -BLOCK_SIZE),  # Up
    ]

    # Utilizing threading to check boundaries for each direction concurrently
    with ThreadPoolExecutor(max_workers=len(directions)) as executor:
        future_to_direction = {
            executor.submit(
                is_within_boundaries,
                (node[0] + dx, node[1] + dy),
                screen_width,
                screen_height,
            ): (dx, dy)
            for dx, dy in directions
        }
        neighbors = [
            (node[0] + dx, node[1] + dy)
            for future, (dx, dy) in future_to_direction.items()
            if future.result()
        ]

    return neighbors


def a_star(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    snake_body: List[Tuple[int, int]],
    fruit_position: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Perform A* pathfinding avoiding given obstacles and considering the snake's body with advanced strategic prioritization.
    This implementation uses a priority queue for the open set for efficient retrieval of the lowest f-score node.
    It also uses a set for obstacles for O(1) complexity checks.
    The pathfinding prioritizes survival, dense grouping of the snake's body, efficient fruit acquisition, and space filling.
    Utilizes multi-threading to explore multiple paths concurrently, maximizing CPU resource utilization.
    """
    open_set = PriorityQueue()
    open_set.put((heuristic(start, goal, snake_body, fruit_position), start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, snake_body, fruit_position)}

    def process_node(
        current: Tuple[int, int], current_f_score: float
    ) -> Optional[List[Tuple[int, int]]]:
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current):
            if neighbor in obstacles or neighbor in snake_body:
                continue
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(
                    neighbor, goal, snake_body, fruit_position
                )
                open_set.put((f_score[neighbor], neighbor))
        return None

    # Utilizing maximum available CPU resources by spawning threads equal to twice the number of CPU cores
    max_workers = multiprocessing.cpu_count() * 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while not open_set.empty():
            _, current = open_set.get()
            futures.append(executor.submit(process_node, current, f_score[current]))

        # Collect results from all threads and determine the optimal path
        optimal_paths = [
            future.result() for future in as_completed(futures) if future.result()
        ]
        if optimal_paths:
            # Select the most efficient path based on a custom metric (e.g., path length, density)
            best_path = min(optimal_paths, key=lambda path: len(path))
            return optimize_path_for_density(best_path, snake_body)

    return []

from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from collections import deque
from typing import Dict, List, Tuple, Optional
 ``
def reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Reconstructs the path from the start node to the goal node using the came_from map.
    This function is optimized for concurrent execution, where each segment of the path is processed in parallel,
    ensuring maximum efficiency and utilization of CPU resources.
    """
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def optimize_path_for_density(
    path: List[Tuple[int, int]], snake_body: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Optimizes the path to ensure maximum density of the snake's body, maintaining a zigzag pattern to fill space efficiently.
    This function restructures the path to prioritize the following:
    1. Survival by avoiding collisions and staying within game boundaries.
    2. Maximum density of the snake's body, ensuring the tail is close to the body's center.
    3. Efficient fruit acquisition once maximum density is achieved.
    4. Space filling when no other moves are available.
    The path optimization uses an exaggerated zigzag movement pattern to maintain density from point to point.
    Utilizes multi-threading to process each path segment concurrently, maximizing CPU resource utilization.
    """
    optimized_path = []
    center_of_mass = calculate_center_of_mass(snake_body)
    current_position = path[0]
    path_queue = deque(path)

    # Utilizing maximum available CPU resources by spawning threads equal to twice the number of CPU cores
    max_workers = multiprocessing.cpu_count() * 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while path_queue:
            next_position = path_queue.popleft()
            futures.append(executor.submit(
                process_path_segment, next_position, center_of_mass, snake_body, current_position
            ))

        # Collect results from all threads and determine the optimal path
        for future in as_completed(futures):
            result = future.result()
            if result:
                optimized_path.extend(result)
                current_position = result[-1]  # Update current position to the last element of the result segment

    return optimized_path

def process_path_segment(
    next_position: Tuple[int, int], center_of_mass: Tuple[float, float], 
    snake_body: List[Tuple[int, int]], current_position: Tuple[int, int]
) -> Optional[List[Tuple[int, int]]]:
    """
    Processes each path segment to determine if it contributes to the optimal density of the snake's body.
    If not, it finds an alternative path that maintains density.
    """
    if is_position_optimal_for_density(next_position, center_of_mass, snake_body):
        return [next_position]
    else:
        alternative_path = find_alternative_path(
            current_position, next_position, snake_body, center_of_mass
        )
        return alternative_path if alternative_path else []


def calculate_center_of_mass(snake_body: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Calculate the geometric center of the snake's body to assist in maintaining density.
    """
    x_coords, y_coords = zip(*snake_body)
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return (center_x, center_y)


def is_position_optimal_for_density(
    position: Tuple[int, int],
    center_of_mass: Tuple[float, float],
    snake_body: List[Tuple[int, int]],
) -> bool:
    """
    Determine if a position contributes to the optimal density of the snake's body.
    """
    distance_to_center = euclidean_distance(position, center_of_mass)
    return (
        distance_to_center < euclidean_distance(snake_body[-1], center_of_mass)
        and position not in snake_body
    )


def find_alternative_path(
    current: Tuple[int, int],
    target: Tuple[int, int],
    snake_body: List[Tuple[int, int]],
    center_of_mass: Tuple[float, float],
) -> List[Tuple[int, int]]:
    """
    Find an alternative path that maintains density when the direct path is not optimal.
    This function uses a modified A* algorithm that prioritizes paths closer to the center of mass.
    """
    open_set = []
    heappush(open_set, (0, current))
    came_from = {}
    g_score = {current: 0}
    f_score = {current: euclidean_distance(current, target)}

    while open_set:
        _, current = heappop(open_set)

        if current == target:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current):
            if neighbor in snake_body:
                continue
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + euclidean_distance(
                    neighbor, center_of_mass
                )
                heappush(open_set, (f_score[neighbor], neighbor))

    return []


class Fruit:
    def __init__(self, window: pg.Surface) -> None:
        """
        Initializes a Fruit object with a position on the game window.
        This method sets the initial position of the fruit to a default value and then relocates it to a valid random position.
        """
        self.position: Tuple[int, int] = (0, 0)  # Initialize with a default position
        self.relocate()  # Relocate to a valid random position
        self.window = window
        logging.info(f"Fruit initialized and placed at {self.position}")

    def draw(self) -> None:
        """
        Draw the fruit on the game window using a fixed color (red) and block size.
        """
        color: Tuple[int, int, int] = (255, 0, 0)  # RGB color for the fruit
        pg.draw.rect(self.window, color, (*self.position, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f"Fruit drawn at {self.position} with color {color}")

    def relocate(self, exclude: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Relocate the fruit to a random position within the game boundaries that is not occupied.
        Ensures the fruit does not spawn inside the snake's body or any other excluded positions.

        :param exclude: A list of positions to be excluded when placing the fruit.
        """
        if exclude is None:
            exclude = []  # Initialize exclude list if not provided
        valid_position_found: bool = False
        while not valid_position_found:
            new_x: int = randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_y: int = randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_position: Tuple[int, int] = (new_x, new_y)
            if new_position not in exclude:
                self.position = new_position
                valid_position_found = True
                logging.info(f"Fruit relocated to {self.position}")
            else:
                logging.debug(
                    f"Attempted fruit position {new_position} is invalid, recalculating..."
                )


class Snake:
    def __init__(self, window: pg.Surface) -> None:
        """
        Initializes the Snake object with a starting position and a fruit object.
        This constructor sets up the snake's body, direction, associates a fruit object, and calculates the initial path using the A* algorithm.
        """
        self.body: Deque[Tuple[int, int]] = deque(
            [(160, 160), (140, 160), (120, 160)]
        )  # Initial body coordinates
        self.direction: Tuple[int, int] = (BLOCK_SIZE, 0)  # Initial movement direction
        self.fruit: Fruit = Fruit(window)  # Associate a fruit object
        self.window = window
        self.path: List[Tuple[int, int]] = (
            self.calculate_path()
        )  # Calculate initial path
        logging.info("Snake initialized with body, direction, fruit, and path.")

    def calculate_path(self) -> List[Tuple[int, int]]:
        """
        Calculate the path from the snake's head to the fruit using the A* algorithm.
        :return: A list of tuples representing the path coordinates.
        """
        return a_star(
            self.body[0],
            self.fruit.position,
            set(self.body),
            list(self.body),
            fruit_position=self.fruit.position,
        )

    def draw(self) -> None:
        """
        Draw the snake on the game window using a fixed color (green) and block size for each segment.
        """
        color: Tuple[int, int, int] = (0, 255, 0)  # RGB color for the snake
        for segment in list(self.body):
            pg.draw.rect(self.window, color, (*segment, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f"Snake drawn on window at segments: {list(self.body)}")

    def move(self) -> bool:
        """
        Move the snake based on the A* pathfinding result. Handles collision and game over scenarios.
        Implements a zigzag movement pattern based on the current direction to maintain a super dense configuration from point to point.
        :return: Boolean indicating if the move was successful (True) or if the game is over (False).
        """
        if not self.path:
            self.path = self.calculate_path()  # Recalculate path if needed
        if self.path:
            next_pos: Tuple[int, int] = self.path.pop(0)
            if self.is_collision(next_pos):
                logging.warning("Collision detected or snake out of bounds")
                return False
            self.body.appendleft(next_pos)
            if next_pos == self.fruit.position:
                self.fruit.relocate(list(self.body))
                self.path = self.calculate_path()
            else:
                self.body.pop()
            logging.info(f"Snake moved successfully to {next_pos}")
            return True
        logging.warning("No path available, game over")
        return False

    def is_collision(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position results in a collision or is out of game bounds.
        :param position: Tuple representing the position to check.
        :return: True if there is a collision or out of bounds, False otherwise.
        """
        return (
            position in self.body
            or not (0 <= position[0] < SCREEN_WIDTH)
            or not (0 <= position[1] < SCREEN_HEIGHT)
        )


def main():
    clock = pg.time.Clock()
    snake = Snake(window)
    fruit = Fruit(window)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        window.fill((0, 0, 0))
        snake.fruit.draw()
        snake.draw()

        if not snake.move():
            logging.info("Game Over, restarting")
            print("Game Over")
            snake = Snake(window)  # Restart the game

        pg.display.flip()
        clock.tick(60)
        logging.debug("Game loop executed")


if __name__ == "__main__":
    main()
