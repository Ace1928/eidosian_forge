import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import logging
from tkinter import Tk, ttk
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any
from functools import wraps
import threading

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for file paths
SOURCE_DIR = Path("/home/lloyd/Dropbox/To_Sort")
DUPLICATES_DIR = Path("/home/lloyd/duplicates")
SORTED_DIR = Path("/home/lloyd/sorted")
ERRORS_DIR = Path("/home/lloyd/errors")
INDEX_FILE = Path("/home/lloyd/index.json")
ERROR_INDEX_FILE = Path("/home/lloyd/error_index.json")

# Ensure directories exist
for directory in [DUPLICATES_DIR, SORTED_DIR, ERRORS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Async error handler decorator
def async_error_handler(
    func: Callable[..., Coroutine[Any, Any, Any]]
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """
    A decorator that wraps an asynchronous function to handle and log any exceptions that occur during execution.

    Args:
        func (Callable[..., Coroutine[Any, Any, Any]]): The asynchronous function to be wrapped.

    Returns:
        Callable[..., Coroutine[Any, Any, Any]]: The wrapped asynchronous function with error handling.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        The wrapper function that executes the wrapped asynchronous function and handles any exceptions.

        Args:
            *args (Any): Positional arguments to be passed to the wrapped function.
            **kwargs (Any): Keyword arguments to be passed to the wrapped function.

        Returns:
            Any: The result of the wrapped function if no exceptions occur.

        Raises:
            Exception: If an exception occurs during the execution of the wrapped function.
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred in {func.__name__}: {e}", exc_info=True)
            if not ERRORS_DIR.exists():
                ERRORS_DIR.mkdir(parents=True, exist_ok=True)
            error_file = (
                ERRORS_DIR
                / f"{func.__name__}_error_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
            )
            async with aiofiles.open(error_file, "a") as f:
                await f.write(str(e))
            raise

    return wrapper


# GUI Components with extensive documentation and type annotations for maintainability and extensibility
class FileBrowserGUI:
    """
    A sophisticated file browsing GUI facilitating directory selection for file processing.
    Implements forward and back navigation alongside a dynamic dropdown menu for file tree display.
    """

    def __init__(self, root: Tk) -> None:
        """
        Initializes the FileBrowserGUI instance with the provided root Tk instance.

        Args:
            root (Tk): The root Tk instance for the GUI.
        """
        self.root: Tk = root
        self.root.title("File Browser")
        self.current_path: Path = SOURCE_DIR
        self.history: List[Path] = [self.current_path]
        self.history_index: int = 0

        # Navigation Frame
        self.nav_frame: ttk.Frame = ttk.Frame(self.root)
        self.nav_frame.pack(fill="x", padx=5, pady=5)

        # Back Button
        self.back_button: ttk.Button = ttk.Button(
            self.nav_frame, text="Back", command=self.go_back
        )
        self.back_button.pack(side="left", padx=5)

        # Forward Button
        self.forward_button: ttk.Button = ttk.Button(
            self.nav_frame, text="Forward", command=self.go_forward
        )
        self.forward_button.pack(side="left", padx=5)

        # Path Dropdown
        self.path_var: ttk.Combobox = ttk.Combobox(self.nav_frame, width=60)
        self.path_var.pack(fill="x", expand=True)
        self.path_var.bind("<<ComboboxSelected>>", self.path_selected)

        self.update_path_dropdown()

    def update_path_dropdown(self) -> None:
        """
        Dynamically updates the dropdown menu with the current directory's subdirectories and files.
        """
        paths: List[str] = [
            str(p) for p in self.current_path.iterdir() if p.is_dir() or p.is_file()
        ]
        self.path_var["values"] = paths
        if paths:
            self.path_var.current(0)

    def path_selected(self, event: Any) -> None:
        """
        Handles path selection from the dropdown menu, navigating to the selected directory.

        Args:
            event (Any): The event triggered by the path selection.
        """
        selected_path: Path = Path(self.path_var.get())
        if selected_path.is_dir():
            self.navigate_to(selected_path)

    def navigate_to(self, path: Path) -> None:
        """
        Navigates to the specified directory, updating navigation history.

        Args:
            path (Path): The directory path to navigate to.
        """
        self.current_path = path
        self.history = self.history[: self.history_index + 1] + [path]
        self.history_index += 1
        self.update_path_dropdown()

    def go_back(self) -> None:
        """
        Navigates to the previous directory in the navigation history.
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.current_path = self.history[self.history_index]
            self.update_path_dropdown()

    def go_forward(self) -> None:
        """
        Navigates to the next directory in the navigation history, if available.
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_path = self.history[self.history_index]
            self.update_path_dropdown()


def launch_file_browser() -> Path:
    """
    Launches the file browser GUI, returning the selected directory path.

    Returns:
        Path: The selected directory path from the file browser GUI.
    """
    root: Tk = Tk()
    gui: FileBrowserGUI = FileBrowserGUI(root)
    root.mainloop()
    return gui.current_path


# Utility Functions with comprehensive type annotations and detailed documentation for asynchronous file processing
async def calculate_file_hash(file_path: Path) -> str:
    """
    Asynchronously calculates the SHA-256 hash of a file, reading in 64kb chunks.

    Args:
        file_path (Path): The path to the file for which the hash needs to be calculated.

    Returns:
        str: The hexadecimal digest of the calculated SHA-256 hash.
    """
    BUF_SIZE: int = 65536  # Define buffer size for efficient reading
    sha256 = hashlib.sha256()  # Initialize SHA-256 hash object
    async with aiofiles.open(
        file_path, "rb"
    ) as f:  # Asynchronously open file for reading in binary mode
        while True:
            data: bytes = await f.read(BUF_SIZE)  # Asynchronously read a chunk of data
            if not data:
                break  # Exit loop if end of file is reached
            sha256.update(data)  # Update hash object with chunk
    return sha256.hexdigest()  # Return hexadecimal digest of the hash


async def move_file(file_path: Path, destination: Path) -> None:
    """
    Moves a file to a specified destination directory, ensuring the destination exists.

    Args:
        file_path (Path): The path to the file to be moved.
        destination (Path): The destination directory path where the file should be moved.
    """
    if not destination.exists():
        destination.mkdir(
            parents=True, exist_ok=True
        )  # Ensure destination directory exists
    destination_file: Path = (
        destination / file_path.name
    )  # Define destination file path
    shutil.move(str(file_path), str(destination_file))  # Move file to destination


@async_error_handler
async def process_file(
    file_path: Path,
    file_hash: str,
    duplicates_index: Dict[str, List[str]],
    kept_files_index: Dict[str, str],
) -> None:
    """
    Processes a file by checking for duplicates and moving it to the appropriate directory.
    Implements advanced hashing and file management strategies.

    Args:
        file_path (Path): The path to the file to be processed.
        file_hash (str): The calculated hash of the file.
        duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
        kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
    """
    file_extension: str = file_path.suffix  # Extract file extension
    destination_dir: Path = SORTED_DIR / file_extension.strip(
        "."
    )  # Define destination directory based on file extension
    today_date: str = datetime.now().strftime(
        "_%d%m%y"
    )  # Get today's date in specified format
    standardized_name: str = (
        f"{file_hash}{today_date}{file_extension}"  # Construct standardized file name
    )
    destination_file: Path = (
        destination_dir / standardized_name
    )  # Define destination file path

    if file_hash in duplicates_index:
        await move_file(file_path, DUPLICATES_DIR)  # Move file to duplicates directory
        duplicates_index[file_hash].append(
            str(file_path)
        )  # Add file path to duplicates index
    else:
        duplicates_index[file_hash] = [
            str(file_path)
        ]  # Initialize list with file path in duplicates index
        kept_files_index[str(destination_file)] = str(
            file_path
        )  # Add file path to kept files index
        if not destination_dir.exists():
            destination_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure destination directory exists
        shutil.move(str(file_path), str(destination_file))  # Move file to destination


async def process_directory(
    directory: Path,
    duplicates_index: Dict[str, List[str]],
    kept_files_index: Dict[str, str],
) -> None:
    """
    Recursively processes each file in a directory and its subdirectories.
    Employs advanced asynchronous programming techniques for efficiency.

    Args:
        directory (Path): The path to the directory to be processed.
        duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
        kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
    """
    for item in directory.iterdir():  # Iterate over items in directory
        if item.is_file():  # Check if item is a file
            file_hash: str = await calculate_file_hash(item)  # Calculate file hash
            await process_file(
                item, file_hash, duplicates_index, kept_files_index
            )  # Process file
        elif item.is_dir():  # Check if item is a directory
            await process_directory(
                item, duplicates_index, kept_files_index
            )  # Recursively process directory


def run_asyncio_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Runs the provided asyncio event loop indefinitely.

    Args:
        loop (asyncio.AbstractEventLoop): The asyncio event loop to be run.
    """
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def async_main() -> None:
    """
    The main asynchronous function to be executed.
    """
    # Async main function logic here
    pass


def main() -> None:
    """
    The main entry point of the script.
    """
    loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    t: threading.Thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    t.start()

    try:
        asyncio.run_coroutine_threadsafe(async_main(), loop)
        root: Tk = Tk()
        gui: FileBrowserGUI = FileBrowserGUI(root)
        root.mainloop()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join()


if __name__ == "__main__":
    main()
