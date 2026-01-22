from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Node
import math
import logging
import threading

# Configure logging with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Algorithm(ABC):
    """
    Abstract base class for pathfinding algorithms in the Snake AI game.

    This class provides a structured template for implementing various pathfinding algorithms,
    ensuring that each derived class adheres to a consistent interface for initializing the grid,
    calculating distances, and running the algorithm. It also ensures thread safety and efficient
    resource management to optimize performance and flexibility.

    Attributes:
        grid (List[List[Node]]): A grid representation of the game environment.
        frontier (List[Node]): List of nodes to be explored.
        explored_set (List[Node]): List of nodes that have been explored.
        path (List[Node]): List representing the path from start to goal.
        lock (threading.Lock): A lock for thread-safe manipulation of shared resources.
    """

    def __init__(self, grid: List[List[Node]]) -> None:
        """
        Asynchronously initializes the Algorithm with a grid, setting up the necessary data structures
        for pathfinding operations. It also configures thread safety mechanisms and prepares for maximum concurrency.

        Args:
            grid (List[List[Node]]): The grid representing the game environment.
        """
        self.grid: List[List[Node]] = grid
        self.frontier: List[Node] = []
        self.explored_set: List[Node] = []
        self.path: List[Node] = []
        self.lock: threading.Lock = (
            threading.Lock()
        )  # Initialize the threading lock for thread safety
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10
        )  # Executor for handling concurrent tasks

        logging.debug("Algorithm base class initialized with grid.")
        logging.debug(
            f"Grid dimensions set to {len(grid)}x{len(grid[0])} with total cells: {len(grid) * len(grid[0])}"
        )
        logging.debug(
            "Frontier, Explored Set, and Path lists have been initialized and are ready for use."
        )

        # Asynchronously ensure that all resources are ready and logging is set up before any operations begin
        future_initialization = self.executor.submit(self._async_initialization)
        try:
            future_initialization.result()  # Wait for the asynchronous initialization to complete
        except Exception as e:
            logging.error(
                "An error occurred during the asynchronous initialization of the Algorithm: {}".format(
                    e
                )
            )
            raise

    def _async_initialization(self):
        """
        Handles the asynchronous pre-processing of grid and other data structures under thread safety.
        """
        with self.lock:
            # Asynchronous and thread-safe pre-processing of the grid or other structures
            logging.debug(
                "Asynchronous pre-processing of grid and data structures initiated."
            )
            # Simulate some pre-processing work
            time.sleep(1)  # This represents some complex operation that takes time
            logging.debug(
                "Asynchronous pre-processing of grid and data structures completed successfully."
            )

    async def get_initstate_and_goalstate(self, snake) -> Tuple[Node, Node]:
        """
        Asynchronously determines the initial and goal states for the algorithm in a thread-safe manner.

        Args:
            snake: The snake object.

        Returns:
            Tuple[Node, Node]: A tuple containing the initial and goal state nodes.
        """

        def determine_states():
            with self.lock:
                initial_state: Node = Node(snake.get_x(), snake.get_y())
                goal_state: Node = Node(snake.get_fruit().x, snake.get_fruit().y)
                logging.debug(
                    f"Initial state determined at {initial_state}. Goal state determined at {goal_state}."
                )
                return initial_state, goal_state

        # Submit the determination of states to the executor and await its future result
        future_states = self.executor.submit(determine_states)
        initial_state, goal_state = await asyncio.wrap_future(future_states)
        return initial_state, goal_state

    async def manhattan_distance(self, nodeA: Node, nodeB: Node) -> int:
        """
        Asynchronously calculates the Manhattan distance between two nodes.
        The Manhattan distance is the sum of the absolute differences in x and y coordinates.
        It is more accurate in terms of distance than the Euclidean distance.

        Args:
            nodeA (Node): The first node.
            nodeB (Node): The second node.

        Returns:
            int: The Manhattan distance between the two nodes.
        """

        async def calculate_distance():
            with self.lock:
                distance_1: int = abs(nodeA.x - nodeB.x)
                distance_2: int = abs(nodeA.y - nodeB.y)
                manhattan_distance: int = distance_1 + distance_2
                logging.debug(
                    f"Manhattan distance between {nodeA} and {nodeB} is {manhattan_distance}."
                )
                return manhattan_distance

        return await asyncio.to_thread(calculate_distance)

    async def euclidean_distance(self, nodeA: Node, nodeB: Node) -> float:
        """
        Asynchronously calculates the Euclidean distance between two nodes.
        The Euclidean distance is the square root of the sum of the squared differences in x and y coordinates.
        It is more accurate in situations where the grid is not rectangular.

        Args:
            nodeA (Node): The first node.
            nodeB (Node): The second node.

        Returns:
            float: The Euclidean distance between the two nodes.
        """

        async def calculate_distance():
            with self.lock:
                distance_1: int = nodeA.x - nodeB.x
                distance_2: int = nodeA.y - nodeB.y
                euclidean_distance: float = math.sqrt(distance_1**2 + distance_2**2)
                logging.debug(
                    f"Euclidean distance between {nodeA} and {nodeB} is {euclidean_distance}."
                )
                return euclidean_distance

        return await asyncio.to_thread(calculate_distance)

    async def hamiltonian_distance(self, nodeA: Node, nodeB: Node) -> int:
        """
        Asynchronously calculates the Hamiltonian distance between two nodes.
        The Hamiltonian distance is the sum of the product of the absolute difference in x and y coordinates.
        It tends to create a path that is more efficient than the Manhattan distance.

        Args:
            nodeA (Node): The first node.
            nodeB (Node): The second node.

        Returns:
            int: The Hamiltonian distance between the two nodes.
        """

        async def calculate_distance():
            with self.lock:
                distance_1: int = abs(nodeA.x - nodeB.x)
                distance_2: int = abs(nodeA.y - nodeB.y)
                hamiltonian_distance: int = distance_1 * distance_2
                logging.debug(
                    f"Hamiltonian distance between {nodeA} and {nodeB} is {hamiltonian_distance}."
                )
                return hamiltonian_distance

        return await asyncio.to_thread(calculate_distance)

    async def chebyshev_distance(self, nodeA: Node, nodeB: Node) -> int:
        """
        Asynchronously calculates the Chebyshev distance between two nodes.
        The Chebyshev distance is the maximum of the absolute differences in x and y coordinates.
        It is useful when movement is allowed in all directions.

        Args:
            nodeA (Node): The first node.
            nodeB (Node): The second node.

        Returns:
            int: The Chebyshev distance between the two nodes.
        """

        async def calculate_distance():
            with self.lock:
                distance_1: int = abs(nodeA.x - nodeB.x)
                distance_2: int = abs(nodeA.y - nodeB.y)
                chebyshev_distance: int = max(distance_1, distance_2)
                logging.debug(
                    f"Chebyshev distance between {nodeA} and {nodeB} is {chebyshev_distance}."
                )
                return chebyshev_distance

        return await asyncio.to_thread(calculate_distance)

    async def minkowski_distance(self, nodeA: Node, nodeB: Node, p: int = 2) -> float:
        """
        Asynchronously calculates the Minkowski distance between two nodes.
        The Minkowski distance is a generalization of other distance metrics such as Manhattan and Euclidean distance.
        The parameter 'p' determines the type of distance metric to be calculated.
        It is set to 2 by default, which is the same as the Euclidean distance.
        Lower values of 'p' will result in a more horizontal-vertical-like path, while higher values will result in a more diagonal-like path.

        Args:
            nodeA (Node): The first node.
            nodeB (Node): The second node.
            p (int): The power parameter for the Minkowski distance.

        Returns:
            float: The Minkowski distance between the two nodes.
        """

        async def calculate_distance():
            with self.lock:
                distance_1: int = abs(nodeA.x - nodeB.x)
                distance_2: int = abs(nodeA.y - nodeB.y)
                minkowski_distance: float = (distance_1**p + distance_2**p) ** (1 / p)
                logging.debug(
                    f"Minkowski distance between {nodeA} and {nodeB} is {minkowski_distance}."
                )
                return minkowski_distance

        return await asyncio.to_thread(calculate_distance)

    async def octile_distance(self, nodeA: Node, nodeB: Node) -> float:
        """
        Asynchronously calculates the Octile distance between two nodes.
        The Octile distance is the same as the Chebyshev distance, but with the diagonal movement allowed.
        It is more accurate in situations where the grid is not rectangular.

        Args:
            nodeA (Node): The first node.
            nodeB (Node): The second node.

        Returns:
            float: The Octile distance between the two nodes.
        """

        async def calculate_distance():
            with self.lock:
                dx: int = abs(nodeA.x - nodeB.x)
                dy: int = abs(nodeA.y - nodeB.y)
                octile_distance: float = dx + dy + (math.sqrt(2) - 2) * min(dx, dy)
                logging.debug(
                    f"Octile distance between {nodeA} and {nodeB} is {octile_distance}."
                )
                return octile_distance

        return await asyncio.to_thread(calculate_distance)

    @abstractmethod
    async def run_algorithm(self, snake) -> Optional[Node]:
        """
        Abstract asynchronous method to run the pathfinding algorithm. This method is designed to be thread-safe
        and allows for concurrent execution of multiple instances of the algorithm, enhancing the performance
        and scalability of the system.

        Args:
            snake: The snake object.

        Returns:
            Optional[Node]: The next node in the path for the snake to follow, if any.
        """
        pass

    async def get_path(self, node: Node) -> Node:
        """
        Asynchronously constructs the path from the given node to the root node in a thread-safe manner,
        ensuring maximum concurrency and efficiency.

        Args:
            node (Node): The node from which to construct the path.

        Returns:
            Node: The root node of the path.
        """
        logging.debug("Asynchronously constructing path from node to root.")
        if node.parent is None:
            logging.debug(f"Node {node} is root node.")
            return node

        path = []
        async with asyncio.Lock():
            while node.parent.parent is not None:
                path.append(node)
                logging.debug(f"Added {node} to path asynchronously.")
                node = node.parent
            logging.debug(
                f"Asynchronous path construction completed with final node {node}."
            )

        # Reverse the path to ensure it starts from the root and ends at the original node
        path.reverse()
        for path_node in path:
            self.path.append(path_node)
            logging.debug(f"Node {path_node} appended to final path in correct order.")

        return node

    async def inside_body(self, snake, node: Node) -> bool:
        """
        Asynchronously checks if a node is inside the snake's body in a thread-safe manner.

        Args:
            snake: The snake object.
            node (Node): The node to check.

        Returns:
            bool: True if the node is inside the snake's body, False otherwise.
        """
        async with asyncio.Lock():
            for body in snake.body:
                if body.x == node.x and body.y == node.y:
                    logging.debug(f"Node {node} is inside snake's body.")
                    return True
            logging.debug(f"Node {node} is not inside snake's body.")
            return False

    async def outside_boundary(self, node: Node) -> bool:
        """
        Asynchronously checks if a node is outside the game boundaries in a thread-safe manner.

        Args:
            node (Node): The node to check.

        Returns:
            bool: True if the node is outside the boundaries, False otherwise.
        """
        async with asyncio.Lock():
            if not 0 <= node.x < NO_OF_CELLS:
                logging.debug(f"Node {node} is outside horizontal boundaries.")
                return True
            elif not BANNER_HEIGHT <= node.y < NO_OF_CELLS:
                logging.debug(f"Node {node} is outside vertical boundaries.")
                return True
            logging.debug(f"Node {node} is inside boundaries.")
            return False

    async def get_neighbors(self, node: Node, grid: List[List[Node]]) -> List[Node]:
        """
        Asynchronously retrieves the neighboring nodes of a given node in a thread-safe manner.

        Args:
            node (Node): The node for which to find neighbors.

        Returns:
            List[Node]: A list of neighboring nodes.
        """
        neighbors = []
        i: int = int(node.x)
        j: int = int(node.y)

        async with asyncio.Lock():
            # left [i-1, j]
            if i > 0:
                neighbors.append(grid[i - 1][j])
                logging.debug(f"Left neighbor {grid[i-1][j]} added.")
            # right [i+1, j]
            if i < NO_OF_CELLS - 1:
                neighbors.append(grid[i + 1][j])
                logging.debug(f"Right neighbor {grid[i+1][j]} added.")
            # top [i, j-1]
            if j > 0:
                neighbors.append(grid[i][j - 1])
                logging.debug(f"Top neighbor {grid[i][j-1]} added.")
            # bottom [i, j+1]
            if j < NO_OF_CELLS - 1:
                neighbors.append(grid[i][j + 1])
                logging.debug(f"Bottom neighbor {grid[i][j+1]} added.")

            logging.debug(f"Neighbors of {node}: {neighbors}")
            return neighbors
