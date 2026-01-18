import pygame
import sys
import math
import logging  # Import the logging module
from typing import List, Tuple  # Import typing for type hints
import pygame.gfxdraw  # For anti-aliased circles

################################################################################
# Logging Configuration
################################################################################
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("Starting the Triangular Numbers Visualizer module...")

################################################################################
# Configuration Class
################################################################################


class Config:
    """
    Configuration class to hold all game settings.

    This class encapsulates various parameters that control the behavior and appearance
    of the Triangular Numbers Visualizer. It promotes easy tweaking and reuse of settings.

    Attributes:
        WINDOW_WIDTH (int): The width of the Pygame window in pixels.
        WINDOW_HEIGHT (int): The height of the Pygame window in pixels.
        WINDOW_CAPTION (str): The caption/text at the top of the window.
        BG_COLOR (tuple): The background color in RGB.
        FONT_NAME (str): Name of the font used for text display.
        FONT_SIZE (int): Size of the font.
        FONT_COLOR (tuple): Color of the text in RGB.
        SHAPE_RADIUS (int): Radius of the circles drawn to represent numbers.
        SHAPE_GLOW_STEPS (int): Number of glow layers.
        COLOR_CYCLE_SPEED (float): Color change speed.
        MIN_SATURATION (int): Minimum color saturation (0-100).
        MAX_SATURATION (int): Maximum color saturation.
        MIN_VALUE (int): Minimum color brightness.
        MAX_VALUE (int): Maximum color brightness.
        STEP_DELAY_MS (int): Delay in milliseconds between adding the next integer.
        MAX_TRIANGULAR_INDEX (int): The maximum n for T_n before resetting.
        TOP_MARGIN (int): The vertical margin from the top of the window.
        LEFT_MARGIN (int): The horizontal margin from the left of the window.
        ROW_SPACING (int): Vertical spacing (in pixels) between rows of circles.
        COL_SPACING (int): Horizontal spacing (in pixels) between circles.
        INFINITE (bool): Whether the sequence continues indefinitely.
    """

    def __init__(self):
        """Initialize the configuration with default values."""
        logging.info("Initializing Config class...")  # Log config initialization

        # Window settings
        self.WINDOW_WIDTH: int = 1000  # Width of the game window in pixels
        self.WINDOW_HEIGHT: int = 700  # Height of the game window in pixels
        self.WINDOW_CAPTION: str = (
            "Triangular Numbers Visualizer - Fun with Math!"  # Caption of the game window
        )
        self.INITIAL_WINDOW_WIDTH: int = (
            self.WINDOW_WIDTH
        )  # Store initial width for scaling
        self.INITIAL_WINDOW_HEIGHT: int = (
            self.WINDOW_HEIGHT
        )  # Store initial height for scaling

        # Background color (R, G, B) - Dark gray for better contrast
        self.BG_COLOR: Tuple[int, int, int] = (30, 30, 30)

        # Text settings
        self.FONT_NAME: str = "Arial"  # Font to be used for text rendering
        self.FONT_SIZE: int = 24  # Size of the font
        self.FONT_COLOR: Tuple[int, int, int] = (
            255,
            255,
            255,
        )  # Color of the text (white)

        # Shape settings - Settings for the circles representing numbers
        self.SHAPE_RADIUS: int = 20  # Radius of each circle shape
        self.SHAPE_GLOW_STEPS: int = 3  # Number of glow layers
        self.COLOR_CYCLE_SPEED: float = 0.1  # Color change speed
        self.MIN_SATURATION: int = 80  # Minimum color saturation (0-100)
        self.MAX_SATURATION: int = 100  # Maximum color saturation
        self.MIN_VALUE: int = 80  # Minimum color brightness
        self.MAX_VALUE: int = 100  # Maximum color brightness

        # Animation settings - Control the pace and extent of the visualization
        self.STEP_DELAY_MS: int = (
            1000  # Delay in milliseconds between adding each number (Shorter delay for faster animation)
        )
        self.MAX_TRIANGULAR_INDEX: int = (
            1000  # Maximum triangular number index to visualize before resetting
        )
        self.TOP_MARGIN: int = 100  # Top margin for the text - moved text to top
        self.LEFT_MARGIN: int = 100  # Initial left margin, will be dynamically adjusted
        self.ROW_SPACING: float = math.sqrt(3) * self.SHAPE_RADIUS * 1.2
        self.COL_SPACING: float = 2 * self.SHAPE_RADIUS * 1.2
        self.INFINITE: bool = True  # Continue indefinitely

        self.BASE_SHAPE_RADIUS = 20  # Reference size for all scaling
        self.BASE_FONT_SIZE = 24
        self.BASE_ROW_SPACING = math.sqrt(3) * self.BASE_SHAPE_RADIUS * 1.2
        self.BASE_COL_SPACING = 2 * self.BASE_SHAPE_RADIUS * 1.2
        self.BASE_TOP_MARGIN = 100

        logging.info("Config class initialized.")  # Log config initialization complete


################################################################################
# Math Utilities
################################################################################


def triangular_number(n: int) -> int:
    """
    Computes the n-th Triangular number.

    The n-th triangular number is the sum of the first n natural numbers, calculated as:
    T_n = 1 + 2 + 3 + ... + n = n * (n + 1) // 2

    :param n: The index of the triangular number to compute (integer).
    :return: The n-th triangular number (integer).
    """
    if not isinstance(n, int):  # Type checking for input
        raise TypeError(f"Input n must be an integer, but got {type(n)}")
    if n < 0:  # Input validation for non-negative index
        raise ValueError(f"Input n must be non-negative, but got {n}")

    result = (n * (n + 1)) // 2  # Calculate triangular number
    logging.debug(f"Triangular number T_{n} calculated as {result}")  # Log calculation
    return result


################################################################################
# Triangular Numbers Visualizer
################################################################################


class TriangularNumbersGame:
    """
    Pygame-based visualizer for step-by-step construction of triangular numbers.

    This class uses Pygame to create a visual representation of how triangular numbers are formed.
    It shows each step of adding the next integer to form T_n and highlights the relationship
    between triangular numbers and the Fibonacci sequence (T_n = F_{n+2} - 1).

    Attributes:
        config (Config): A configuration object with all the necessary settings.
        screen (pygame.Surface): The main Pygame surface for drawing.
        font (pygame.font.Font): The font used to render text.
        current_n (int): The current triangular number index being built.
        current_step (int): The step within the current triangular number row.
        last_step_time (int): Timestamp (in ms) of the last integer added (for animation).
        fib_sequence (list): Pre-generated list of Fibonacci numbers for reference.
        color_phase (float): For color cycling.
        current_scale (float): Scaling factor for dynamic scaling.
    """

    def __init__(self, config: Config = Config()):
        """
        Initialize the Triangular Numbers Game.

        :param config: Configuration object containing game settings (Config, optional, default Config()).
        """
        logging.info("Initializing TriangularNumbersGame...")  # Log game initialization
        print("Starting Triangular Numbers Visualizer!")  # Informative print statement

        self.config: Config = config  # Store the configuration
        self.initial_config = Config()  # Store initial config for scaling

        # Pygame setup
        logging.info("Initializing Pygame...")  # Log Pygame initialization
        pygame.init()  # Initialize Pygame modules
        self.screen: pygame.Surface = pygame.display.set_mode(
            (self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT), pygame.RESIZABLE
        )  # Create the display screen, resizable window
        pygame.display.set_caption(self.config.WINDOW_CAPTION)  # Set the window caption
        logging.info("Pygame initialized, display created.")  # Log Pygame display setup

        # Font for rendering text
        self.font: pygame.font.Font = pygame.font.SysFont(
            self.config.FONT_NAME, self.config.FONT_SIZE
        )  # Load system font
        logging.info(
            f"Font loaded: {self.config.FONT_NAME}, size: {self.config.FONT_SIZE}"
        )  # Log font loading

        # Game state variables
        self.current_n: int = (
            1  # Current triangular number index being built (starts from T_1)
        )
        self.current_step: int = (
            1  # Current step within T_n (up to which integer we've displayed in the nth row)
        )
        self.last_step_time: int = (
            pygame.time.get_ticks()
        )  # Time of the last step update
        logging.debug(
            f"Initial game state: current_n={self.current_n}, current_step={self.current_step}"
        )  # Log initial game state

        self.color_phase: float = 0.0  # For color cycling
        self.current_scale: float = 1.0

        logging.info(
            "TriangularNumbersGame initialized."
        )  # Log game initialization complete
        print(
            "Ready to visualize triangular numbers and Fibonacci connection!"
        )  # Informative print

    def run(self) -> None:
        """
        Main game loop for the Triangular Numbers Visualizer.

        This method starts the Pygame event loop, which handles events, updates game logic,
        and draws each frame until the user quits.
        """
        logging.info("Starting game loop...")  # Log game loop start
        print("Starting the visualizer...")  # Informative print

        clock: pygame.time.Clock = (
            pygame.time.Clock()
        )  # Pygame clock to control frame rate

        while True:  # Main game loop
            clock.tick(60)  # Limit frame rate to 60 FPS
            self._handle_events()  # Process events (user input, system events)
            self._update_logic()  # Update game state and logic
            self._draw_frame()  # Render the current frame

        logging.info(
            "Game loop finished (should not reach here unless error)."
        )  # Log game loop end

    ############################################################################
    # Internal Game Logic
    ############################################################################

    def _handle_events(self) -> None:
        """
        Handles Pygame events.

        Processes events such as user input (keyboard, mouse) and system events (quit).
        Currently, only handles the QUIT event to close the application and window resize events.
        """
        for event in pygame.event.get():  # Get all events from the event queue
            if event.type == pygame.QUIT:  # Check if the user clicked the close button
                logging.info("QUIT event received. Exiting game.")  # Log quit event
                print(
                    "Thank you for exploring Triangular Numbers!"
                )  # Kid-friendly exit message
                pygame.quit()  # Uninitialize Pygame modules
                sys.exit()  # Exit the Python program
            if event.type == pygame.VIDEORESIZE:  # Handle window resize event
                logging.info(
                    f"VIDEORESIZE event received, new size: {event.size}"
                )  # Log resize event
                self._handle_resize(event.size)  # Call resize handler

    def _handle_resize(self, new_size: Tuple[int, int]) -> None:
        """Simplified resize handler"""
        self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT = new_size
        self.screen = pygame.display.set_mode(new_size, pygame.RESIZABLE)

    def _calculate_dynamic_scale(self, n: int) -> float:
        """Dynamic scaling based on triangle size and window dimensions"""
        triangle_width = n * self.config.BASE_COL_SPACING
        triangle_height = n * self.config.BASE_ROW_SPACING + self.config.BASE_TOP_MARGIN

        width_scale = self.config.WINDOW_WIDTH / (triangle_width * 1.1)
        height_scale = self.config.WINDOW_HEIGHT / (triangle_height * 1.1)
        return min(width_scale, height_scale, 1.0)

    def _hsv_to_rgb(
        self, hue: float, saturation: int, value: int
    ) -> Tuple[int, int, int]:
        """Convert HSV to RGB with neon effect"""
        saturation = max(
            self.config.MIN_SATURATION, min(saturation, self.config.MAX_SATURATION)
        )
        value = max(self.config.MIN_VALUE, min(value, self.config.MAX_VALUE))
        color = pygame.Color(0)
        color.hsva = (hue % 360, saturation, value, 100)
        return (color.r, color.g, color.b)

    def _update_logic(self) -> None:
        """
        Updates the game logic based on time and game state.

        This method controls the step-by-step construction of triangular numbers.
        It increments 'current_step' to add the next circle in the current row (n),
        and moves to the next triangular number (increment 'current_n') when a row is complete.
        """
        now: int = pygame.time.get_ticks()  # Get current time in milliseconds
        if (
            now - self.last_step_time > self.config.STEP_DELAY_MS
        ):  # Check if enough time has passed since the last step
            logging.debug("Time for next step update.")  # Log step update trigger

            self.current_step += (
                1  # Increment step to add the next circle in the current row
            )
            logging.debug(
                f"Current step incremented to {self.current_step}"
            )  # Log step increment
            print(
                f"Adding circle number {self.current_step} in row T_{self.current_n}..."
            )

            if (
                self.current_step > self.current_n
            ):  # Check if the current row (n) is complete
                logging.debug(f"Row {self.current_n} complete.")  # Log row completion
                self.current_n += 1  # Move to the next triangular number index
                self.current_step = 1  # Reset step to 1 for the new row
                logging.info(
                    f"Moving to next triangular number T_{self.current_n}."
                )  # Log move to next T_n

                if (
                    not self.config.INFINITE
                    and self.current_n > self.config.MAX_TRIANGULAR_INDEX
                ):  # Check if max index is exceeded and not infinite
                    logging.info(
                        f"Reached MAX_TRIANGULAR_INDEX ({self.config.MAX_TRIANGULAR_INDEX}). Resetting to T_1."
                    )  # Log max index reached
                    print(
                        f"Reached max T_{self.config.MAX_TRIANGULAR_INDEX}; resetting to T_1."
                    )  # Kid-friendly reset message
                    self.current_n = 1  # Reset to start from T_1

            self.last_step_time = now  # Update last step time
            logging.debug(
                f"Last step time updated to {self.last_step_time}"
            )  # Log last step time update

        self.color_phase = (self.color_phase + self.config.COLOR_CYCLE_SPEED) % 360

    def get_scale_factor(self) -> float:
        """Calculate unified scale factor based on window size and triangle dimensions"""
        base_width = self.initial_config.INITIAL_WINDOW_WIDTH
        base_height = self.initial_config.INITIAL_WINDOW_HEIGHT
        return min(
            self.config.WINDOW_WIDTH / base_width,
            self.config.WINDOW_HEIGHT / base_height,
            1.0,
        )

    def _draw_frame(self) -> None:
        """Updated drawing with unified scaling"""
        self.screen.fill(self.config.BG_COLOR)

        # Calculate base scale factor
        base_scale = self.get_scale_factor()
        dynamic_scale = self._calculate_dynamic_scale(self.current_n)
        combined_scale = min(base_scale, dynamic_scale)

        # Scale all elements
        current_font_size = int(self.config.BASE_FONT_SIZE * combined_scale)
        current_radius = int(self.config.BASE_SHAPE_RADIUS * combined_scale)
        row_spacing = self.config.BASE_ROW_SPACING * combined_scale
        col_spacing = self.config.BASE_COL_SPACING * combined_scale
        top_margin = self.config.BASE_TOP_MARGIN * combined_scale

        # Update font temporarily
        temp_font = pygame.font.SysFont(self.config.FONT_NAME, current_font_size)

        # Draw text and get height
        text_height = self._draw_text_info(temp_font, top_margin, combined_scale)

        # Draw triangle with calculated scaling
        self._draw_triangular_shapes(
            top_offset=top_margin + text_height + 20 * combined_scale,
            row_spacing=row_spacing,
            col_spacing=col_spacing,
            radius=current_radius,
            scale=combined_scale,
        )

        pygame.display.flip()

    def _draw_text_info(
        self, font: pygame.font.Font, top_margin: float, scale: float
    ) -> int:
        """Scaled text drawing"""
        text_color = self._hsv_to_rgb(self.color_phase, 100, 100)
        lines = [
            f"Building Triangular Number T_{self.current_n}",
            f"Current Step: {self.current_step}/{self.current_n}",
            f"Partial Sum: {triangular_number(self.current_n-1) + self.current_step}",
            f"Projected T_{self.current_n}: {triangular_number(self.current_n)}",
            "Watch the triangle grow step by step!",
        ]

        y_offset = top_margin
        x_offset = 10 * scale
        total_height: float = 0

        for text in lines:
            surface = font.render(text, True, text_color)
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += font.get_height() + 5 * scale
            total_height += font.get_height() + 5 * scale

        return int(total_height)

    def _draw_triangular_shapes(
        self,
        top_offset: float,
        row_spacing: float,
        col_spacing: float,
        radius: int,
        scale: float,
    ):
        """Unified scaled drawing"""
        max_row_width = self.current_n * col_spacing
        start_x = (self.config.WINDOW_WIDTH - max_row_width) // 2

        for row in range(1, self.current_n + 1):
            circles = row if row < self.current_n else self.current_step
            base_hue = (self.color_phase + row * 15) % 360

            for col in range(circles):
                x = int(
                    start_x
                    + col * col_spacing
                    + (self.current_n - row) * (col_spacing // 2)
                )
                y = int(top_offset + (row - 1) * row_spacing)

                hue = (base_hue + col * 5) % 360
                main_color = self._hsv_to_rgb(hue, 100, 100)

                # Glow effect
                for i in range(self.config.SHAPE_GLOW_STEPS, 0, -1):
                    alpha = int(50 / i)
                    glow_radius = radius + i * 2
                    pygame.gfxdraw.filled_circle(
                        self.screen, x, y, glow_radius, (*main_color, alpha)
                    )

                pygame.draw.circle(self.screen, main_color, (x, y), radius)


################################################################################
# Main Execution
################################################################################


def main() -> None:
    """
    Main function to run the Triangular Numbers Visualizer.

    Creates an instance of the TriangularNumbersGame and starts the game loop.
    """
    logging.info("Starting main function.")  # Log main function start
    print("Initializing the Triangular Numbers Visualizer...")  # Start message

    config = Config()  # Create configuration object
    game: TriangularNumbersGame = TriangularNumbersGame(
        config
    )  # Create the game instance
    logging.info(
        "TriangularNumbersGame instance created."
    )  # Log game instance creation
    print("Game instance created. Starting the run loop.")  # Game start message
    game.run()  # Start the game loop
    logging.info("Main function finished.")  # Log main function end
    print("Exiting the Triangular Numbers Visualizer.")  # Exit message


if __name__ == "__main__":
    main()  # Execute main function when script is run directly
