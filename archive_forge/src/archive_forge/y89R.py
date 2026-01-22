import logging
from collections import deque
from typing import Deque, List, Optional
from Utility import Node
from Algorithm import Algorithm

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BFS(Algorithm):
    def __init__(self, grid: List[List[int]]) -> None:
        super().__init__(grid)
        logging.debug("BFS algorithm instance initialized with grid.")

    def run_algorithm(self, snake) -> Optional[Node]:
        logging.debug("Starting BFS algorithm.")
        # Initialize data structures
        self.frontier: Deque[Node] = deque([])
        self.explored_set: List[Node] = []
        self.path: List[Node] = []

        # Retrieve initial and goal states
        initialstate, goalstate = self.get_initstate_and_goalstate(snake)
        logging.debug(
            f"Initial state set to {initialstate}, goal state set to {goalstate}."
        )

        # Begin with the initial state
        self.frontier.append(initialstate)
        logging.debug(f"Initial state {initialstate} added to frontier.")

        # Process nodes until the frontier is empty
        while len(self.frontier) > 0:
            shallowest_node = self.frontier.popleft()  # FIFO queue
            self.explored_set.append(shallowest_node)
            logging.debug(f"Processing node {shallowest_node} from frontier.")

            # Retrieve neighbors
            neighbors = self.get_neighbors(shallowest_node)
            logging.debug(f"Neighbors retrieved: {neighbors}")

            # Evaluate each neighbor
            for neighbor in neighbors:
                # Check conditions for skipping the neighbor
                if self.inside_body(snake, neighbor) or self.outside_boundary(neighbor):
                    self.explored_set.append(neighbor)
                    logging.debug(
                        f"Skipping neighbor {neighbor} due to collision or boundary."
                    )
                    continue

                # Check if the neighbor is new
                if neighbor not in self.frontier and neighbor not in self.explored_set:
                    neighbor.parent = shallowest_node  # Link to parent
                    self.explored_set.append(neighbor)  # Mark as visited
                    self.frontier.append(neighbor)  # Add to frontier
                    logging.debug(
                        f"Neighbor {neighbor} added to frontier with parent {shallowest_node}."
                    )

                    # Check if the goal state is reached
                    if neighbor.equal(goalstate):
                        logging.debug("Goal state reached.")
                        path_to_return = self.get_path(neighbor)
                        logging.debug(f"Path found: {path_to_return}")
                        return path_to_return

        # If no path is found
        logging.debug("No path found to goal state.")
        return None
