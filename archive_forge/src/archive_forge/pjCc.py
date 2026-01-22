import time
import pygame as pg
from pygame.math import Vector2
from typing import List, Set, Tuple, Optional
from random import randint

# Constants for the maze simulation
MAZE_WIDTH = 40
MAZE_HEIGHT = 20
NUM_SCENARIOS = 10
SCENARIO_DURATION = 5  # seconds
OBSTACLE_MIN = 50
OBSTACLE_MAX = 150
SCREEN_MARGIN = 0.1  # 10% margin on each side

# Colors
COLOR_BACKGROUND = (255, 255, 255)  # White
COLOR_OBSTACLE = (0, 0, 0)  # Black
COLOR_START = (0, 255, 0)  # Green
COLOR_GOAL = (255, 0, 0)  # Red
COLOR_PATH = (0, 0, 255)  # Blue


class Node:
    def __init__(self, x: int, y: int):
        self.position: Vector2 = Vector2(x, y)
        self.parent: Optional["Node"] = None
        self.f: float = 0.0
        self.g: float = 0.0
        self.h: float = 0.0

    def is_equal(self, other: "Node") -> bool:
        return self.position == other.position


class SearchAlgorithm:
    def __init__(self, snake, apple):
        self.snake = snake
        self.apple = apple

    def refresh_maze(self):
        maze = []
        for i in range(20):
            row = []
            for j in range(40):
                row.append(0)
            maze.append(row)
        for i in range(len(self.snake.body)):
            # Ensure that the indices are integers
            maze[int(self.snake.body[i].y)][int(self.snake.body[i].x)] = -1
        head_position = self.snake.get_head_position()
        tail_position = self.snake.get_tail_position()
        maze[int(head_position.y)][int(head_position.x)] = 1
        maze[int(tail_position.y)][int(tail_position.x)] = 2
        return maze

    def get_path(self):
        maze = self.refresh_maze()
        start, end = None, None
        for i in range(40):
            for j in range(20):
                if maze[j][i] == 1:
                    start = {"x": i, "y": j}
                elif maze[j][i] == 2:
                    end = {"x": i, "y": j}
        node_path = self.astar(maze, start, end)
        vector_path = []
        for node in node_path:
            vector_path.append(Vector2(node.position.x, node.position.y))
        self.snake.path = vector_path

    def astar(self, maze, start, end):
        start_node = Node(start["x"], start["y"])
        end_node = Node(end["x"], end["y"])
        open_list = []
        closed_list = []
        open_list.append(start_node)
        possible_paths = []
        adjacent_squares = [
            [0, -1],
            [0, 1],
            [-1, 0],
            [1, 0],
        ]
        while len(open_list) > 0:
            current_node = open_list[0]
            current_index = 0
            index = 0
            for i in range(len(open_list)):
                if open_list[i].f > current_node.f:
                    current_node = open_list[i]
                    current_index = index
                index += 1
            open_list.pop(current_index)
            closed_list.append(current_node)
            if current_node.is_equal(end_node):
                path = []
                current = current_node
                while current is not None:
                    path.append(current)
                    current = current.parent
                possible_paths.append(list(reversed(path)))
            children = []
            for i in range(len(adjacent_squares)):
                node_position = [
                    int(current_node.position.x + adjacent_squares[i][0]),
                    int(current_node.position.y + adjacent_squares[i][1]),
                ]
                if 0 <= node_position[0] <= 39:
                    if 0 <= node_position[1] <= 19:
                        if maze[node_position[1]][node_position[0]] != -1:
                            new_node = Node(node_position[0], node_position[1])
                            children.append(new_node)
            for i in range(len(children)):
                if_in_closed_list = False
                for j in range(len(closed_list)):
                    if children[i].is_equal(closed_list[j]):
                        if_in_closed_list = True
                if not if_in_closed_list:
                    children[i].g = current_node.g + 2
                    children[i].h = abs(
                        children[i].position.x - end_node.position.x
                    ) + abs(children[i].position.y - end_node.position.y)
                    children[i].f = children[i].g + children[i].h
                    present = False
                    for j in range(len(open_list)):
                        if (
                            children[i].is_equal(open_list[j])
                            and children[i].g < open_list[j].g
                        ):
                            present = True
                        elif (
                            children[i].is_equal(open_list[j])
                            and children[i].g >= open_list[j].g
                        ):
                            open_list[j] = children[i]
                            open_list[j].parent = current_node
                    if not present:
                        children[i].parent = current_node
                        open_list.append(children[i])
        path = []
        for i in range(len(possible_paths)):
            if len(possible_paths[i]) > len(path):
                path = possible_paths[i]
        return path


def generate_complex_maze(
    width: int, height: int, num_obstacles: int
) -> List[List[int]]:
    """
    Generate a complex maze with specified dimensions and number of obstacles, ensuring
    that the maze is not only filled with obstacles but also has a structured layout
    resembling walls and paths.

    Args:
        width (int): The width of the maze.
        height (int): The height of the maze.
        num_obstacles (int): The number of obstacles to place in the maze.

    Returns:
        List[List[int]]: A 2D list representing the maze where -1 indicates an obstacle.
    """
    maze = [[0 for _ in range(width)] for _ in range(height)]
    # Create outer walls
    for x in range(width):
        maze[0][x] = maze[height - 1][x] = -1
    for y in range(height):
        maze[y][0] = maze[y][width - 1] = -1

    # Generate internal walls and paths
    obstacle_count = 0
    while obstacle_count < num_obstacles:
        x, y = randint(1, width - 2), randint(1, height - 2)
        if maze[y][x] == 0:  # Ensure not placing an obstacle on another
            # Placing a vertical or horizontal wall
            if randint(0, 1):
                for offset in range(-1, 2):
                    if 0 <= y + offset < height and maze[y + offset][x] == 0:
                        maze[y + offset][x] = -1
                        obstacle_count += 1
            else:
                for offset in range(-1, 2):
                    if 0 <= x + offset < width and maze[y][x + offset] == 0:
                        maze[y][x + offset] = -1
                        obstacle_count += 1

    return maze


def draw_maze(screen: pg.Surface, maze: List[List[int]], block_size: int):
    """
    Draw the maze on the Pygame screen, representing obstacles and free paths.

    Args:
        screen (pg.Surface): The Pygame screen to draw on.
        maze (List[List[int]]): The maze to draw.
        block_size (int): The size of each block in pixels.
    """
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            rect = pg.Rect(x * block_size, y * block_size, block_size, block_size)
            if cell == -1:
                pg.draw.rect(screen, COLOR_OBSTACLE, rect)
            else:
                pg.draw.rect(screen, COLOR_BACKGROUND, rect)


def draw_path(screen: pg.Surface, path: List[Node], block_size: int) -> None:
    """
    Draw the path on the Pygame screen, visualizing the movement of the snake along the path.

    Args:
        screen (pg.Surface): The Pygame screen to draw on.
        path (List[Node]): The path consisting of nodes to draw.
        block_size (int): The size of each block in pixels, defining the size of the drawn rectangle.
    """
    for node in path:
        rect = pg.Rect(
            node.position.x * block_size,
            node.position.y * block_size,
            block_size,
            block_size,
        )
        pg.draw.rect(screen, COLOR_PATH, rect)
        pg.display.flip()
        time.sleep(0.1)  # Visualize each step with a delay


def draw_start_goal(
    screen: pg.Surface, start: Tuple[int, int], goal: Tuple[int, int], block_size: int
):
    """
    Draw the start and goal positions on the Pygame screen, marking the beginning and end of the path.

    Args:
        screen (pg.Surface): The Pygame screen to draw on.
        start (Tuple[int, int]): The starting position (x, y).
        goal (Tuple[int, int]): The goal position (x, y).
        block_size (int): The size of each block in pixels.
    """
    start_rect = pg.Rect(
        start[0] * block_size, start[1] * block_size, block_size, block_size
    )
    goal_rect = pg.Rect(
        goal[0] * block_size, goal[1] * block_size, block_size, block_size
    )
    pg.draw.rect(screen, COLOR_START, start_rect)
    pg.draw.rect(screen, COLOR_GOAL, goal_rect)


def run_scenario(
    screen: pg.Surface,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    maze: List[List[int]],
    block_size: int,
):
    """
    Run a single scenario of pathfinding from start to goal in the given maze, visualizing the snake's path.

    Args:
        screen (pg.Surface): The Pygame screen to draw on.
        start (Tuple[int, int]): The starting position (x, y).
        goal (Tuple[int, int]): The goal position (x, y).
        maze (List[List[int]]): The maze to navigate.
        block_size (int): The size of each block in pixels.
    """
    # Convert start and goal tuples to dictionaries expected by the astar function
    start_dict = {"x": start[0], "y": start[1]}
    goal_dict = {"x": goal[0], "y": goal[1]}

    # Generate a random snake body for the scenario
    snake_body_length = randint(3, (MAZE_WIDTH * MAZE_HEIGHT) // 2)
    snake_body = []
    for _ in range(snake_body_length):
        x, y = randint(0, MAZE_WIDTH - 1), randint(0, MAZE_HEIGHT - 1)
        while maze[y][x] == -1:  # Ensure the body is not placed on an obstacle
            x, y = randint(0, MAZE_WIDTH - 1), randint(0, MAZE_HEIGHT - 1)
        snake_body.append(Vector2(x, y))

    # Create a snake object with the generated body
    snake = type(
        "Snake",
        (object,),
        {
            "body": snake_body,
            "get_head_position": lambda: snake_body[0],
            "get_tail_position": lambda: snake_body[-1],
        },
    )

    # Initialize the search algorithm with the snake and a dummy apple (not used in this scenario)
    search_algorithm = SearchAlgorithm(snake, None)
    search_algorithm.get_path()  # This should internally set up the path correctly

    draw_maze(screen, maze, block_size)
    draw_start_goal(screen, start, goal, block_size)
    pg.display.flip()
    time.sleep(1)  # Pause for a moment to view the maze

    if search_algorithm.snake.path:
        draw_path(screen, search_algorithm.snake.path, block_size)
        pg.display.flip()
        time.sleep(1)  # Pause for a moment to view the path

        for node in search_algorithm.snake.path:
            rect = pg.Rect(
                node.x * block_size,
                node.y * block_size,
                block_size,
                block_size,
            )
            pg.draw.rect(screen, COLOR_PATH, rect)
            pg.display.flip()
            time.sleep(0.1)  # Delay between each step of the path
    else:
        print(f"No path found from {start} to {goal}")


def main():
    """
    Main function to run multiple scenarios of pathfinding in complex mazes using Pygame.
    """
    pg.init()
    screen_info = pg.display.Info()
    screen_width, screen_height = int(screen_info.current_w * 0.8), int(
        screen_info.current_h * 0.8
    )
    screen = pg.display.set_mode((screen_width, screen_height))
    pg.display.set_caption("Maze Pathfinding Simulation")
    clock = pg.time.Clock()

    block_size_x = (screen_width * (1 - 2 * SCREEN_MARGIN)) // MAZE_WIDTH
    block_size_y = (screen_height * (1 - 2 * SCREEN_MARGIN)) // MAZE_HEIGHT
    block_size = min(block_size_x, block_size_y)

    for scenario in range(NUM_SCENARIOS):
        num_obstacles = randint(OBSTACLE_MIN, OBSTACLE_MAX)
        maze = generate_complex_maze(MAZE_WIDTH, MAZE_HEIGHT, num_obstacles)
        start = (randint(0, MAZE_WIDTH - 1), randint(0, MAZE_HEIGHT - 1))
        goal = (randint(0, MAZE_WIDTH - 1), randint(0, MAZE_HEIGHT - 1))

        print(f"Scenario {scenario + 1}:")
        run_scenario(screen, start, goal, maze, block_size)
        time.sleep(SCENARIO_DURATION)

        for event in pg.event.get():
            if event.type is pg.QUIT or (
                event.type is pg.KEYDOWN and event.key == pg.K_ESCAPE
            ):
                pg.quit()
                return

        screen.fill(COLOR_BACKGROUND)
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()
