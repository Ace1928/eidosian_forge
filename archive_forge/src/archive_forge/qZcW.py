def heuristic(
    self,
    a: Tuple[int, int],
    b: Tuple[int, int],
    last_dir: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Calculate the heuristic value for A* algorithm using the Euclidean distance.
    This heuristic is improved by adding a directional bias to discourage straight paths,
    promote zigzagging, and encourage dense packing. It also considers potential escape routes
    and avoids obstacles, boundaries, and the snake's body.

    Args:
        a (Tuple[int, int]): The current node coordinates.
        b (Tuple[int, int]): The goal node coordinates.
        last_dir (Optional[Tuple[int, int]]): The last direction moved.

    Returns:
        float: The computed heuristic value.
    """
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    manhattan_distance = dx + dy

    # Directional Bias: Penalize moving in the same direction to promote zigzagging
    direction_penalty = 0
    if last_dir:
        current_dir = (a[0] - b[0], a[1] - b[1])
        if current_dir == last_dir:
            direction_penalty = 5

    # Boundary Avoidance: Penalize nodes close to the grid boundaries
    boundary_threshold = 2
    boundary_penalty = 0
    if (
        a[0] < boundary_threshold
        or a[0] >= GRID_SIZE - boundary_threshold
        or a[1] < boundary_threshold
        or a[1] >= GRID_SIZE - boundary_threshold
    ):
        boundary_penalty = 10

    # Obstacle Avoidance: Penalize nodes that are adjacent to obstacles
    obstacle_penalty = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = (a[0] + dx, a[1] + dy)
        if neighbor in self.obstacles:
            obstacle_penalty += 5

    # Snake Body Avoidance: Heavily penalize nodes that are part of the snake's body
    snake_body_penalty = 0
    if a in self.snake:
        snake_body_penalty = float("inf")

    # Escape Route: Favor nodes with more available neighboring nodes
    escape_route_bonus = 0
    available_neighbors = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = (a[0] + dx, a[1] + dy)
        if (
            0 <= neighbor[0] < GRID_SIZE
            and 0 <= neighbor[1] < GRID_SIZE
            and neighbor not in self.snake
            and neighbor not in self.obstacles
        ):
            available_neighbors += 1
    escape_route_bonus = available_neighbors * -2

    # Dense Packing: Favor nodes that are closer to other parts of the snake's body
    dense_packing_bonus = 0
    for segment in self.snake:
        dense_packing_bonus += 1 / (
            1 + math.sqrt((a[0] - segment[0]) ** 2 + (a[1] - segment[1]) ** 2)
        )

    # Calculate the final heuristic value
    heuristic_value = (
        manhattan_distance
        + direction_penalty
        + boundary_penalty
        + obstacle_penalty
        + snake_body_penalty
        + escape_route_bonus
        + dense_packing_bonus
    )

    return heuristic_value


def a_star_search(
    self, start: Tuple[int, int], goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Perform the A* search algorithm to find the optimal path from start to goal.
    This implementation uses a priority queue to efficiently explore nodes, a closed set
    to avoid redundant processing, and a custom heuristic function that considers multiple
    factors to generate strategic and efficient paths.

    Args:
        start (Tuple[int, int]): The starting position of the path.
        goal (Tuple[int, int]): The goal position of the path.

    Returns:
        List[Tuple[int, int]]: The optimal path from start to goal as a list of coordinates.
    """
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    g_score = defaultdict(lambda: float("inf"))
    g_score[start] = 0
    f_score = defaultdict(lambda: float("inf"))
    f_score[start] = self.heuristic(start, goal)

    closed_set = set()
    last_direction = None

    while not open_set.empty():
        current = open_set.get()[1]
        closed_set.add(current)

        if current == goal:
            return self.reconstruct_path(came_from, current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < GRID_SIZE
                and 0 <= neighbor[1] < GRID_SIZE
                and neighbor not in self.snake
                and neighbor not in self.obstacles
                and neighbor not in closed_set
            ):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal, (dx, dy)
                    )
                    open_set.put((f_score[neighbor], neighbor))
                    last_direction = (dx, dy)

    return []
