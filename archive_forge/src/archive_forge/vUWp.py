from Algorithm import Algorithm
from typing import List, Optional
import logging

# To avoid side effects of logging configuration on other modules, we ensure that logging is configured
# only when the module is executed as the main module. This encapsulation prevents the logging configurations
# from affecting other modules that might import this module.
if __name__ == "__main__":
    # Configure logging to debug level and specify the format for logging.
    # This configuration is critical for tracing the execution of the algorithm and understanding its flow.
    # The format includes the time of the log entry, the level of severity, and the message.
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class A_STAR(Algorithm):
    def __init__(self, grid: "Grid") -> None:
        # The constructor of the A_STAR class initializes the algorithm with a specific grid.
        # It inherits from the Algorithm class, allowing it to use shared functionality and structure.
        super().__init__(grid)

        # A debug log entry is created when an instance of A_STAR is initialized.
        # This log entry will help in debugging by confirming that the grid has been successfully initialized.
        logging.debug("A_STAR algorithm initialized with grid.")

    def run_algorithm(self, snake: "Snake") -> Optional[List["Node"]]:
        # clear everything
        self.frontier: List["Node"] = []
        self.explored_set: List["Node"] = []
        self.path: List["Node"] = []

        initialstate: "Node"
        goalstate: "Node"
        initialstate, goalstate = self.get_initstate_and_goalstate(snake)

        # open list
        self.frontier.append(initialstate)
        logging.debug(f"Initial state {initialstate} added to frontier.")

        # while we have states in open list
        while len(self.frontier) > 0:
            # get node with lowest f(n)
            lowest_index: int = 0
            for i in range(len(self.frontier)):
                if self.frontier[i].f < self.frontier[lowest_index].f:
                    lowest_index = i

            lowest_node: "Node" = self.frontier.pop(lowest_index)
            logging.debug(f"Lowest node {lowest_node} popped from frontier.")

            # check if it's goal state
            if lowest_node.equal(goalstate):
                logging.info("Goal state reached.")
                return self.get_path(lowest_node)

            self.explored_set.append(lowest_node)  # mark visited
            logging.debug(f"Node {lowest_node} added to explored set.")
            neighbors: List["Node"] = self.get_neighbors(lowest_node)  # get neighbors

            # for each neighbor
            for neighbor in neighbors:
                # check if path inside snake, outside boundary or already visited
                if (
                    self.inside_body(snake, neighbor)
                    or self.outside_boundary(neighbor)
                    or neighbor in self.explored_set
                ):
                    logging.debug(
                        f"Skipping neighbor {neighbor} due to invalid conditions."
                    )
                    continue  # skip this path

                g: int = lowest_node.g + 1
                best: bool = False  # assuming neighbor path is better

                if neighbor not in self.frontier:  # first time visiting
                    neighbor.h = self.manhattan_distance(goalstate, neighbor)
                    self.frontier.append(neighbor)
                    best = True
                    logging.debug(
                        f"Neighbor {neighbor} added to frontier with heuristic {neighbor.h}."
                    )
                elif lowest_node.g < neighbor.g:  # has already been visited
                    best = True  # but had a worse g now its better

                if best:
                    neighbor.parent = lowest_node
                    neighbor.g = g
                    neighbor.f = neighbor.g + neighbor.h
                    logging.debug(f"Updated neighbor {neighbor} with new g, f values.")
        logging.info("No path found to goal state.")
        return None
