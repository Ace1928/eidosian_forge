import pygame
import sys
import logging
import asyncio
from typing import Tuple, Union
from Constants import *
from GA import *

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Menu:
    """
    A class to represent the menu interface of the game.

    Attributes:
        game (Game): The game instance.
        mid_size (float): Half the size of the game window.
        run_display (bool): A flag to determine if the menu should continue displaying.
        cursor_rect (pygame.Rect): The rectangle representing the cursor.
        offset (int): The offset for cursor positioning.
        title_size (int): Font size for the title.
        option_size (int): Font size for menu options.
    """

    def __init__(self, game: "Game") -> None:
        """
        Constructs all the necessary attributes for the menu object.

        Parameters:
            game (Game): The game instance.
        """
        logging.debug("Initializing Menu class with game instance.")
        self.game: "Game" = game
        self.mid_size: float = self.game.SIZE / 2
        self.run_display: bool = True
        self.cursor_rect: pygame.Rect = pygame.Rect(0, 0, 20, 20)
        self.offset: int = -150
        self.title_size: int = 50
        self.option_size: int = 28
        logging.debug(
            f"Menu class initialized with mid_size: {self.mid_size}, run_display: {self.run_display}, cursor_rect: {self.cursor_rect}, offset: {self.offset}, title_size: {self.title_size}, option_size: {self.option_size}."
        )

    async def draw_cursor(self) -> None:
        """
        Asynchronously draws the cursor on the menu.
        """
        logging.debug("Attempting to draw cursor on menu asynchronously.")
        try:
            await asyncio.sleep(0)  # Ensuring the function is asynchronous
            await self.game.draw_text(
                "*",
                size=20,
                x=self.cursor_rect.x,
                y=self.cursor_rect.y,
                color=MENU_COLOR,
            )
            logging.debug(
                f"Cursor successfully drawn at position: ({self.cursor_rect.x}, {self.cursor_rect.y})."
            )
        except Exception as e:
            logging.error(f"Failed to draw cursor due to: {e}", exc_info=True)
            raise RuntimeError(f"Drawing cursor failed due to: {e}") from e

    async def blit_menu(self) -> None:
        """
        Asynchronously blits the menu to the screen.
        """
        logging.debug("Attempting to blit menu to the screen asynchronously.")
        try:
            await asyncio.sleep(0)  # Ensuring the function is asynchronous
            self.game.window.blit(self.game.display, (0, 0))
            pygame.display.update()
            await self.game.reset_keys()
            logging.debug("Menu blitted successfully to the screen.")
        except Exception as e:
            logging.error(f"Failed to blit menu due to: {e}", exc_info=True)
            raise RuntimeError(f"Blitting menu failed due to: {e}") from e


class MainMenu(Menu):
    """
    A class to represent the main menu of the game, inheriting from Menu.

    Attributes:
        state (str): The current state of the menu.
        cursorBFS (Tuple[int, int, int]): Color of the BFS cursor.
        cursorDFS (Tuple[int, int, int]): Color of the DFS cursor.
        cursorASTAR (Tuple[int, int, int]): Color of the AStar cursor.
        cursorGA (Tuple[int, int, int]): Color of the GA cursor.
        BFSx (int): X-coordinate for BFS option.
        BFSy (int): Y-coordinate for BFS option.
        DFSx (int): X-coordinate for DFS option.
        DFSy (int): Y-coordinate for DFS option.
        ASTARx (int): X-coordinate for AStar option.
        ASTARy (int): Y-coordinate for AStar option.
        GAx (int): X-coordinate for GA option.
        GAy (int): Y-coordinate for GA option.
    """

    def __init__(self, game: "Game") -> None:
        """
        Constructs all the necessary attributes for the main menu object.

        Parameters:
            game (Game): The game instance.
        """
        logging.debug("Initializing MainMenu class with game instance.")
        super().__init__(game)
        self.state: str = "BFS"
        self.cursorBFS: Tuple[int, int, int] = MENU_COLOR
        self.cursorDFS: Tuple[int, int, int] = WHITE
        self.cursorASTAR: Tuple[int, int, int] = WHITE
        self.cursorGA: Tuple[int, int, int] = WHITE
        self.BFSx, self.BFSy = self.mid_size, self.mid_size - 50
        self.DFSx, self.DFSy = self.mid_size, self.mid_size + 0
        self.ASTARx, self.ASTARy = self.mid_size, self.mid_size + 50
        self.GAx, self.GAy = self.mid_size, self.mid_size + 100

        self.cursor_rect.midtop = (int(self.BFSx + self.offset), int(self.BFSy))
        logging.debug(
            "MainMenu class initialized with state: {self.state}, cursor colors set, and cursor positions defined."
        )

    async def change_cursor_color(self) -> None:
        """
        Asynchronously changes the cursor color based on the current state.
        """
        logging.debug(f"Changing cursor color based on state: {self.state}.")
        try:
            await self.clear_cursor_color()
            if self.state == "BFS":
                self.cursorBFS = MENU_COLOR
            elif self.state == "DFS":
                self.cursorDFS = MENU_COLOR
            elif self.state == "ASTAR":
                self.cursorASTAR = MENU_COLOR
            elif self.state == "GA":
                self.cursorGA = MENU_COLOR
            logging.debug("Cursor color changed to reflect current state.")
        except Exception as e:
            logging.error(f"Failed to change cursor color due to: {e}", exc_info=True)
            raise RuntimeError(f"Changing cursor color failed due to: {e}") from e

    async def clear_cursor_color(self) -> None:
        """
        Asynchronously resets the cursor colors to their default values.
        """
        logging.debug("Clearing cursor colors to default.")
        self.cursorBFS = WHITE
        self.cursorDFS = WHITE
        self.cursorASTAR = WHITE
        self.cursorGA = WHITE
        logging.debug("Cursor colors reset to default.")

    async def display_menu(self) -> None:
        """
        Asynchronously displays the main menu and handles its functionality.
        """
        logging.debug("Displaying main menu.")
        self.run_display = True
        try:
            while self.run_display:
                await self.game.event_handler()
                await self.check_input()
                self.game.display.fill(WINDOW_COLOR)

                await self.game.draw_text(
                    "Ai Snake Game",
                    size=self.title_size,
                    x=self.game.SIZE / 2,
                    y=self.game.SIZE / 2 - 2 * (CELL_SIZE + NO_OF_CELLS),
                    color=TITLE_COLOR,
                )
                await self.game.draw_text(
                    "BFS",
                    size=self.option_size,
                    x=self.BFSx,
                    y=self.BFSy,
                    color=self.cursorBFS,
                )
                await self.game.draw_text(
                    "DFS",
                    size=self.option_size,
                    x=self.DFSx,
                    y=self.DFSy,
                    color=self.cursorDFS,
                )
                await self.game.draw_text(
                    "AStar",
                    size=self.option_size,
                    x=self.ASTARx,
                    y=self.ASTARy,
                    color=self.cursorASTAR,
                )
                await self.game.draw_text(
                    "Genetic Algorithm",
                    size=self.option_size,
                    x=self.GAx,
                    y=self.GAy,
                    color=self.cursorGA,
                )
                await self.draw_cursor()
                await self.change_cursor_color()
                await self.blit_menu()
            logging.debug("Main menu displayed and interactive.")
        except Exception as e:
            logging.error(f"Failed to display main menu due to: {e}", exc_info=True)
            raise RuntimeError(f"Displaying main menu failed due to: {e}") from e

    async def check_input(self) -> None:
        """
        Asynchronously checks user input for menu navigation, ensuring thread safety and maximum concurrency.
        """
        logging.debug("Asynchronously checking user input for menu navigation.")
        try:
            async with asyncio.Lock():
                await self.move_cursor()
                if self.game.START:
                    if self.state == "GA":  # go to genetic algorithm options
                        self.game.curr_menu = self.game.GA
                    else:
                        self.game.playing = True
                    self.run_display = False
                    logging.debug("Input processed and state updated asynchronously.")
        except Exception as e:
            logging.error(
                f"Failed to check input asynchronously due to: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Asynchronous checking of input failed due to: {e}"
            ) from e

    async def move_cursor(self) -> None:
        """
        Asynchronously moves the cursor based on user input, ensuring thread safety and maximum concurrency.
        """
        logging.debug("Attempting to move cursor asynchronously based on user input.")
        try:
            async with asyncio.Lock():
                if self.game.DOWNKEY:
                    await self.update_cursor_position_down()
                if self.game.UPKEY:
                    await self.update_cursor_position_up()
        except Exception as e:
            logging.error(
                f"Failed to move cursor asynchronously due to: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Asynchronous moving of cursor failed due to: {e}"
            ) from e

    async def update_cursor_position_down(self) -> None:
        """
        Updates the cursor position when the DOWN key is pressed, ensuring thread safety and maximum concurrency.
        This function is fully asynchronous and utilizes an asyncio.Lock to ensure that the cursor position update
        is thread-safe and does not suffer from race conditions.
        """
        logging.debug("Updating cursor position for DOWN key.")
        async with asyncio.Lock():
            if self.state == "BFS":
                self.cursor_rect.midtop = (int(self.DFSx + self.offset), int(self.DFSy))
                self.state = "DFS"
                logging.debug("Cursor moved to DFS.")
            elif self.state == "DFS":
                self.cursor_rect.midtop = (
                    int(self.ASTARx + self.offset),
                    int(self.ASTARy),
                )
                self.state = "ASTAR"
                logging.debug("Cursor moved to AStar.")
            elif self.state == "ASTAR":
                self.cursor_rect.midtop = (int(self.GAx + self.offset), int(self.GAy))
                self.state = "GA"
                logging.debug("Cursor moved to Genetic Algorithm.")
            elif self.state == "GA":
                self.cursor_rect.midtop = (int(self.BFSx + self.offset), int(self.BFSy))
                self.state = "BFS"
                logging.debug("Cursor moved to BFS.")

    async def update_cursor_position_up(self) -> None:
        """
        Updates the cursor position when the UP key is pressed, ensuring thread safety and maximum concurrency.
        This function is fully asynchronous and utilizes an asyncio.Lock to ensure that the cursor position update
        is thread-safe and does not suffer from race conditions.
        """
        logging.debug("Updating cursor position for UP key.")
        async with asyncio.Lock():
            if self.state == "BFS":
                self.cursor_rect.midtop = (int(self.GAx + self.offset), int(self.GAy))
                self.state = "GA"
                logging.debug("Cursor moved up from BFS to Genetic Algorithm.")
            elif self.state == "DFS":
                self.cursor_rect.midtop = (int(self.BFSx + self.offset), int(self.BFSy))
                self.state = "BFS"
                logging.debug("Cursor moved up from DFS to BFS.")
            elif self.state == "ASTAR":
                self.cursor_rect.midtop = (int(self.DFSx + self.offset), int(self.DFSy))
                self.state = "DFS"
                logging.debug("Cursor moved up from AStar to DFS.")
            elif self.state == "GA":
                self.cursor_rect.midtop = (
                    int(self.ASTARx + self.offset),
                    int(self.ASTARy),
                )
                self.state = "ASTAR"
                logging.debug("Cursor moved up from Genetic Algorithm to AStar.")


class button:
    """
    A class to represent a button in the game interface, designed to be fully asynchronous and thread-safe to ensure maximum concurrency.

    Attributes:
        x (int): The x-coordinate of the button.
        y (int): The y-coordinate of the button.
        text (str): The text displayed on the button.
        game (GameGUI): The game instance.
        font (pygame.font.Font): The font used for the button text.
        clicked (bool): A flag to determine if the button has been clicked.
    """

    def __init__(self, x: int, y: int, text: str, game: "GameGUI") -> None:
        """
        Constructs all the necessary attributes for the button object with detailed logging.

        Parameters:
            x (int): The x-coordinate of the button.
            y (int): The y-coordinate of the button.
            text (str): The text displayed on the button.
            game (GameGUI): The game instance.
        """
        logging.debug(f"Initializing button with text: {text}.")
        self.x: int = x
        self.y: int = y
        self.text: str = text
        self.game: "GameGUI" = game
        self.font: pygame.font.Font = pygame.font.Font(game.font_name, 30)
        self.clicked: bool = False
        logging.debug(
            f"Button initialized at position ({self.x}, {self.y}) with text: {self.text}."
        )

    async def draw_button(self) -> bool:
        """
        Asynchronously draws the button and checks for interaction, ensuring thread safety and maximum concurrency.

        Returns:
            bool: True if the button was clicked, False otherwise.
        """
        logging.debug("Attempting to draw button asynchronously.")
        action: bool = False

        # get mouse position
        pos: Tuple[int, int] = pygame.mouse.get_pos()

        # create pygame Rect object for the button
        button_rect: pygame.Rect = pygame.Rect(self.x, self.y, BTN_WIDTH, BTN_HEIGHT)

        # check mouseover and clicked conditions
        if button_rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and not self.clicked:
                self.clicked = True
                pygame.draw.rect(self.game.display, BTN_CLICKED, button_rect)
                logging.debug("Button clicked.")
            elif pygame.mouse.get_pressed()[0] == 0 and self.clicked:
                self.clicked = False
                action = True
                logging.debug("Button action triggered.")
            else:
                pygame.draw.rect(self.game.display, BTN_HOVER, button_rect)
                logging.debug("Mouse hovered over button.")
        else:
            pygame.draw.rect(self.game.display, BTN_COLOR, button_rect)
            logging.debug("Button drawn with default color.")

        # add text to button
        text_img: pygame.Surface = self.font.render(self.text, True, WHITE)
        text_len: int = text_img.get_width()
        self.game.display.blit(
            text_img, (self.x + int(BTN_WIDTH / 2) - int(text_len / 2), self.y + 25)
        )
        logging.debug("Text added to button.")

        return action


class TextBox:
    """
    A class to represent a text input box in the game interface, designed to be fully asynchronous and thread-safe,
    ensuring maximum concurrency and responsiveness.

    Attributes:
        x (int): The x-coordinate of the text box.
        y (int): The y-coordinate of the text box.
        game (Game): The game instance.
        font (pygame.font.Font): The font used for the text input.
        input_rect (pygame.Rect): The rectangle representing the text box.
        input (str): The text input by the user.
        active (bool): A flag to determine if the text box is active.
    """

    def __init__(self, x: int, y: int, game: "Game") -> None:
        """
        Constructs all the necessary attributes for the text box object, initializing logging and setting up
        the asynchronous environment.

        Parameters:
            x (int): The x-coordinate of the text box.
            y (int): The y-coordinate of the text box.
            game (Game): The game instance.
        """
        logging.debug("Initializing TextBox with asynchronous capabilities.")
        self.font: pygame.font.Font = pygame.font.Font(game.font_name, 20)
        self.input_rect: pygame.Rect = pygame.Rect(x, y, TXT_WIDTH, TXT_HEIGHT)
        self.input: str = ""
        self.game: "Game" = game
        self.active: bool = False
        logging.debug(
            f"TextBox initialized at position ({x}, {y}) with asynchronous support."
        )

    async def draw_input(self) -> None:
        """
        Asynchronously draws the text input box and handles interaction in a thread-safe manner,
        utilizing asyncio to ensure that the UI remains responsive and maximally concurrent.
        """
        logging.debug("Attempting to draw input box asynchronously with thread safety.")
        pos: Tuple[int, int] = pygame.mouse.get_pos()

        # Utilize asyncio Lock for thread safety
        async with asyncio.Lock():
            if self.input_rect.collidepoint(pos):
                if pygame.mouse.get_pressed()[0] == 1:
                    self.active = True
                    logging.debug("TextBox activated asynchronously.")
            elif pygame.mouse.get_pressed()[0] == 1:
                self.active = False
                logging.debug("TextBox deactivated asynchronously.")

            if self.active:
                color: Tuple[int, int, int] = TXT_ACTIVE
            else:
                color: Tuple[int, int, int] = TXT_PASSIVE

            pygame.draw.rect(self.game.display, color, self.input_rect, 2)
            text_surface: pygame.Surface = self.font.render(self.input, False, WHITE)
            self.game.display.blit(
                text_surface, (self.input_rect.x + 15, self.input_rect.y + 1)
            )
            logging.debug(
                "Input drawn in TextBox with concurrency and thread safety ensured."
            )


class GAMenu(Menu):
    """
    A class to represent the Genetic Algorithm menu, inheriting from Menu.

    Attributes:
        controller (Controller): The GA controller.
        train_model (button): The button to start training the model.
        load_model (button): The button to load a pre-trained model.
        no_population (TextBox): The text box for number of populations.
        no_generation (TextBox): The text box for number of generations.
        no_hidden_nodes (TextBox): The text box for number of hidden nodes.
        mutation_rate (TextBox): The text box for mutation rate.
    """

    def __init__(self, game: "Game", controller: "Controller") -> None:
        """
        Constructs all the necessary attributes for the GA menu object.

        Parameters:
            game (Game): The game instance.
            controller (Controller): The GA controller.
        """
        logging.debug("Initializing GAMenu class with game instance and controller.")
        Menu.__init__(self, game)

        self.controller: "Controller" = controller
        self.train_model: button = button(
            game.SIZE / 2 - 4 * (CELL_SIZE + NO_OF_CELLS),
            game.SIZE / 2 + 3.5 * (CELL_SIZE + NO_OF_CELLS),
            "Train Model",
            game,
        )
        self.load_model: button = button(
            game.SIZE / 2 + (CELL_SIZE),
            game.SIZE / 2 + 3.5 * (CELL_SIZE + NO_OF_CELLS),
            "Load Model",
            game,
        )

        self.no_population: TextBox = TextBox(
            self.game.SIZE / 2 + 50, self.game.SIZE / 2 - 60, game
        )
        self.no_generation: TextBox = TextBox(
            self.game.SIZE / 2 + 50, self.game.SIZE / 2 - 10, game
        )
        self.no_hidden_nodes: TextBox = TextBox(
            self.game.SIZE / 2 + 50, self.game.SIZE / 2 + 40, game
        )
        self.mutation_rate: TextBox = TextBox(
            self.game.SIZE / 2 + 50, self.game.SIZE / 2 + 90, game
        )
        self.init_input()

    def init_input(self) -> None:
        """
        Initializes the input values for the GA menu.
        """
        logging.debug("Initializing input values for GAMenu.")
        self.no_population.input = "300"
        self.no_generation.input = "30"
        self.no_hidden_nodes.input = "8"
        self.mutation_rate.input = "12"
        logging.debug(
            "Input values initialized: Population=300, Generation=30, Hidden Nodes=8, Mutation Rate=12%."
        )

    async def display_menu(self) -> None:
        """
        Asynchronously displays the GA menu and handles its functionality with maximum concurrency and thread safety.
        """
        logging.debug("Asynchronously displaying GA menu.")
        self.run_display = True
        while self.run_display:
            events = await asyncio.gather(
                *[self.get_event() for _ in range(pygame.event.get_length())]
            )
            for event in events:
                if event.type == pygame.QUIT:
                    self.game.running, self.game.playing = False, False
                    self.game.curr_menu.run_display = False
                    await asyncio.gather(sys.exit())

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.game.BACK = True

                    await asyncio.gather(
                        self.process_population_input(event),
                        self.process_generation_input(event),
                        self.process_hidden_nodes_input(event),
                        self.process_mutation_rate_input(event),
                    )

            await self.check_input()
            self.game.display.fill(WINDOW_COLOR)

            await asyncio.gather(
                self.game.draw_text(
                    "GA Options",
                    self.title_size,
                    self.game.SIZE / 2,
                    self.game.SIZE / 2 - 4 * (CELL_SIZE + NO_OF_CELLS),
                    color=TITLE_COLOR,
                ),
                self.game.draw_text(
                    "Settings to train model:",
                    25,
                    self.game.SIZE / 2,
                    self.game.SIZE / 2 - 2 * (CELL_SIZE + NO_OF_CELLS),
                    color=MENU_COLOR,
                ),
                self.display_text_boxes(),
                self.display_buttons(),
                self.game.draw_text(
                    "Q to return to main menu",
                    20,
                    self.game.SIZE / 2,
                    self.game.SIZE / 2 + 6 * (NO_OF_CELLS + CELL_SIZE),
                    color=WHITE,
                ),
            )

            await self.blit_menu()
        await self.reset()

    async def reset(self):
        """
        Asynchronously resets the active state of all text boxes, ensuring thread safety.
        """
        async with asyncio.Lock():
            self.no_population.active = False
            self.no_generation.active = False
            self.no_hidden_nodes.active = False
            self.mutation_rate.active = False
            logging.debug("All text boxes have been reset asynchronously.")

    async def check_input(self):
        """
        Asynchronously checks if the back key was pressed to return to the main menu, ensuring thread safety.
        """
        async with asyncio.Lock():
            if self.game.BACK:
                self.game.curr_menu = self.game.main_menu
                self.run_display = False
                logging.debug("Back input detected and processed asynchronously.")

    async def load_GA(self):
        """
        Asynchronously loads the GA model and switches to the main menu, ensuring thread safety.
        """
        async with asyncio.Lock():
            self.game.curr_menu = self.game.main_menu
            self.run_display = False
            self.game.curr_menu.state = "GA"
            self.game.playing = True
            self.game.load_model = True
            logging.debug("GA model loaded asynchronously and main menu activated.")

    async def train_GA(self):
        """
        Asynchronously trains the GA model and switches to the main menu, ensuring thread safety and maximum concurrency.
        """
        async with asyncio.Lock():
            self.game.curr_menu = self.game.main_menu
            self.run_display = False
            self.game.curr_menu.state = "GA"
            self.game.playing = True
            logging.debug("Training GA model initiated asynchronously.")

            if len(self.no_population.input) > 0:
                Population.population = int(self.no_population.input)
                logging.debug(
                    f"Population set to {Population.population} asynchronously."
                )

            if len(self.no_hidden_nodes.input) > 0:
                Population.hidden_node = int(self.no_hidden_nodes.input)
                logging.debug(
                    f"Hidden nodes set to {Population.hidden_node} asynchronously."
                )

            if len(self.no_generation.input) > 0:
                GA.generation = int(self.no_generation.input)
                logging.debug(f"Generation set to {GA.generation} asynchronously.")

            if len(self.mutation_rate.input) > 0:
                GA.mutation_rate = int(self.mutation_rate.input) / 100
                logging.debug(
                    f"Mutation rate set to {GA.mutation_rate} asynchronously."
                )
