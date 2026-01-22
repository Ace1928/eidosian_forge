from typing import Optional, Tuple, List, Union, Any
import logging
import asyncio
from Snake import Snake
from Constants import NO_OF_CELLS, BANNER_HEIGHT
from Utility import Grid, Node
from DFS import DFS
from BFS import BFS
from A_STAR import A_STAR
from GA import GA, Population
from pygame.math import Vector2
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GameController:
    """
    The GameController class orchestrates the overall game mechanics, including the management of the snake(s),
    the game grid, and the application of various AI algorithms to control the snake's behavior.
    """

    def __init__(self) -> None:
        """
        Initializes the GameController with default values for its properties.
        """
        logging.debug("Initializing GameController")
        self.snake: Optional[Snake] = None
        self.snakes: List[Snake] = []
        self.score: int = 0
        self.end: bool = False
        self.grid: List[List[Node]] = Grid().grid
        self.algo: Optional[Union[GA, BFS, DFS, A_STAR]] = None
        self.model_loaded: bool = False
        logging.debug("GameController initialized with empty parameters")

    async def reset(self) -> None:
        """
        Resets the game controller to its initial state, clearing any active game data.
        """
        logging.debug("Resetting GameController state")
        self.end = False
        if self.snake:
            await self.snake.reset()
            self.snake = None

        self.algo = None
        self.snakes = []
        self.model_loaded = False
        logging.debug("GameController state reset complete")

    async def best_GA_score(self) -> int:
        """
        Retrieves the best score achieved by a genetic algorithm-controlled snake.

        Returns:
            int: The best score, or 0 if no genetic algorithm is active.
        """
        logging.debug("Fetching best GA score")
        return (
            await self.algo.best_score if self.algo and isinstance(self.algo, GA) else 0
        )

    async def best_GA_gen(self) -> int:
        """
        Retrieves the generation number of the best-scoring genetic algorithm-controlled snake.

        Returns:
            int: The generation number, or 0 if no genetic algorithm is active.
        """
        logging.debug("Fetching best GA generation")
        return (
            await self.algo.best_gen if self.algo and isinstance(self.algo, GA) else 0
        )

    async def curr_gen(self) -> int:
        """
        Retrieves the current generation number of the genetic algorithm.

        Returns:
            int: The current generation number, or 0 if no genetic algorithm is active.
        """
        logging.debug("Fetching current GA generation")
        return (
            await self.algo.generation if self.algo and isinstance(self.algo, GA) else 0
        )

    async def save_model(self) -> None:
        """
        Saves the model of the best-performing snake if the current algorithm is a genetic algorithm.
        """
        logging.debug("Saving model")
        if isinstance(self.algo, GA):
            best_snake = await self.algo.best_snake
            if best_snake:
                network = await best_snake.network
                await best_snake.save_model(network, "saved_model")
                logging.debug("Model saved successfully")
            else:
                logging.error("No best snake to save model from")
        else:
            logging.error("Algorithm is not GA, cannot save model")

    async def load_model(self) -> None:
        """
        Loads a saved model into the snake if available.
        """
        logging.debug("Loading model")
        self.snake = Snake()
        await self.snake.load_model("saved_model")
        self.model_loaded = True
        logging.debug("Model loaded successfully")

    async def get_score(self) -> int:
        """
        Retrieves the current score of the active snake.

        Returns:
            int: The current score, or 0 if no snake is active.
        """
        logging.debug("Fetching score")
        return await self.snake.score if self.snake else 0

    async def ate_fruit(self) -> None:
        """
        Checks if the snake has eaten a fruit and updates the game state accordingly.
        """
        logging.debug("Checking if snake ate fruit")
        if self.snake and await self.snake.ate_fruit():
            await self.snake.add_body_ai()
            await self.change_fruit_location()
            logging.debug("Snake ate fruit and body was added")

    async def change_fruit_location(self) -> None:
        """
        Changes the location of the fruit on the grid, ensuring it does not spawn inside the snake's body.
        """
        logging.debug("Changing fruit location")
        while True:
            await self.snake.create_fruit()
            position: Vector2 = await self.snake.get_fruit()
            inside_body: bool = any(position == body for body in self.snake.body)

            if not inside_body:
                logging.debug(f"Fruit placed at {position}")
                break
            else:
                logging.debug(f"Collision detected at {position}, retrying")

    async def ate_fruit_GA(self, snake: Snake) -> None:
        """
        Checks if a genetic algorithm-controlled snake has eaten a fruit and updates the game state accordingly.

        Args:
            snake (Snake): The genetic algorithm-controlled snake to check.
        """
        logging.debug(f"Checking if GA snake {snake} ate fruit")
        if await snake.ate_fruit():
            await snake.add_body_ai()
            await self.change_fruit_location_GA(snake)
            logging.debug("GA snake ate fruit and body was added")

    async def change_fruit_location_GA(self, snake: Snake) -> None:
        """
        Changes the location of the fruit for a genetic algorithm-controlled snake, ensuring it does not spawn inside the snake's body.

        Args:
            snake (Snake): The genetic algorithm-controlled snake for which to change the fruit location.
        """
        logging.debug(f"Changing fruit location for GA snake {snake}")
        while True:
            await snake.create_fruit()
            position: Vector2 = await snake.get_fruit()
            inside_body: bool = any(position == body for body in snake.body)

            if not inside_body:
                logging.debug(f"Fruit placed at {position} for GA snake {snake}")
                break
            else:
                logging.debug(
                    f"Collision detected at {position} for GA snake {snake}, retrying"
                )

    async def died(self) -> None:
        """
        Checks if the active snake has died by moving out of bounds or eating its own body.
        """
        logging.debug("Checking if snake has died")
        current_x: int = self.snake.body[0].x
        current_y: int = self.snake.body[0].y

        if not 0 <= current_x < NO_OF_CELLS:
            self.end = True
            logging.debug("Snake died by moving out of horizontal bounds")
        elif not BANNER_HEIGHT <= current_y < NO_OF_CELLS:
            self.end = True
            logging.debug("Snake died by moving out of vertical bounds")
        elif await self.snake.ate_body():
            self.end = True
            logging.debug("Snake died by eating its own body")

    async def get_fruit_pos(self) -> Vector2:
        """
        Retrieves the current position of the fruit on the grid.

        Returns:
            Vector2: The position of the fruit.
        """
        logging.debug("Fetching fruit position")
        return await self.snake.get_fruit()

    async def set_algorithm(self, algo_type: str) -> None:
        """
        Sets the algorithm used to control the snake's movement based on the specified type.

        Args:
            algo_type (str): The type of algorithm to set ('BFS', 'DFS', 'ASTAR', 'GA').
        """
        logging.debug(f"Setting algorithm type to {algo_type}")
        if self.algo is not None:
            logging.debug("Algorithm already set, skipping")
            return

        if algo_type == "BFS":
            self.algo = BFS(self.grid)
            self.snake = Snake()
            logging.debug("BFS algorithm set")

        elif algo_type == "DFS":
            self.algo = DFS(self.grid)
            self.snake = Snake()
            logging.debug("DFS algorithm set")

        elif algo_type == "ASTAR":
            self.algo = A_STAR(self.grid)
            self.snake = Snake()
            logging.debug("A* algorithm set")

        elif algo_type == "GA":
            self.algo = GA(self.grid)
            if not self.model_loaded:
                await self.algo.population.initialize_population()
                self.snakes = self.algo.population.snakes
                logging.debug("GA algorithm set and initial population created")

    async def ai_play(self, algorithm: str) -> None:
        """
        Initiates AI-controlled gameplay using the specified algorithm.

        Args:
            algorithm (str): The type of algorithm to use for AI gameplay ('BFS', 'DFS', 'ASTAR', 'GA').
        """
        logging.debug(f"AI play initiated with algorithm {algorithm}")
        await self.set_algorithm(algorithm)

        if self.algo is None:
            logging.error("Algorithm not set, aborting AI play")
            return

        if isinstance(self.algo, GA):
            await self.update_GA_ai()
        else:
            pos: Optional[Node] = await self.algo.run_algorithm(self.snake)
            await self.update_path_finding_algo(pos)

    def keep_moving(self) -> Tuple[int, int]:
        """
        Continues the snake's movement in the last known direction.

        Returns:
            Tuple[int, int]: The new position of the snake's head.
        """
        logging.debug("Continuing movement based on last direction")
        x: int = self.snake.body[0].x
        y: int = self.snake.body[0].y

        if self.snake.body[1].x == x:
            if self.snake.body[1].y < y:
                # keep going down
                y = y + 1
            else:
                # keep going up
                y = y - 1
        elif self.snake.body[1].y == y:
            if self.snake.body[1].x < x:
                # keep going right
                x = x + 1
            else:
                # keep going left
                x = x - 1
        logging.debug(f"New position determined: ({x}, {y})")
        return x, y

    def update_GA_ai(self) -> None:
        """
        Updates the state of the game when using a genetic algorithm for AI control.
        """
        logging.debug("Updating GA AI")
        if not self.snake and not self.model_loaded:
            if self.algo.done():
                if self.algo.next_generation():
                    self.snakes = self.algo.population.snakes
                    logging.debug("Moved to next generation in GA")
                else:
                    self.end = True
                    logging.debug("GA algorithm ended, no more generations")

            for snake in self.snakes:
                x, y = self.algo.run_algorithm(snake)

                snake.move_ai(x, y)
                self.algo.died(snake)
                self.ate_fruit_GA(snake)
        else:
            x, y = self.algo.run_algorithm(self.snake)
            self.snake.move_ai(x, y)
            self.died()
            self.ate_fruit()

    def update_path_finding_algo(self, pos: Optional[Node]) -> None:
        """
        Updates the state of the game when using a path-finding algorithm for AI control.

        Args:
            pos (Optional[Node]): The next position for the snake to move to, if available.
        """
        logging.debug("Updating path-finding algorithm")
        if pos is None:
            x, y = self.keep_moving()
        else:
            x = pos.x
            y = pos.y

        self.snake.move_ai(x, y)
        self.died()
        self.ate_fruit()
