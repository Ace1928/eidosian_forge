#!/home/lloyd/indego/bin/python

import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import logging
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any, Optional
from functools import wraps
import threading
import ctypes
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QTreeView,
    QFileSystemModel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)
from PyQt5.QtCore import QDir, QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow
import os
import ctypes

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


class FileOperationWorker(QObject):
    """
    Asynchronously handles file operations including hashing, detecting duplicates,
    and organizing files into directories based on their hash values and extensions.

    Attributes:
        directory (Path): The root directory for file operations.
        duplicates_index (Dict[str, List[str]]): Tracks duplicate files by their hash.
        kept_files_index (Dict[str, str]): Tracks files that have been processed without duplication.
        progress (pyqtSignal): Signal to report operation progress.
        finished (pyqtSignal): Signal to indicate completion of operations.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(
        self,
        directory: Path,
        duplicates_index: Dict[str, List[str]],
        kept_files_index: Dict[str, str],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.directory: Path = directory
        self.duplicates_index: Dict[str, List[str]] = duplicates_index
        self.kept_files_index: Dict[str, str] = kept_files_index

    async def calculate_file_hash(self, file_path: Path) -> str:
        """
        Asynchronously calculates the SHA-256 hash of a file.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The hex digest of the file hash.
        """
        BUF_SIZE = 65536  # Read in 64kb chunks
        sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            while True:
                data = await f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

    async def move_file(self, file_path: Path, destination: Path) -> None:
        """
        Asynchronously moves a file to a specified destination directory.

        Args:
            file_path (Path): The path to the file to be moved.
            destination (Path): The target directory path.
        """
        try:
            if not destination.exists():
                destination.mkdir(parents=True, exist_ok=True)
                destination_file = destination / file_path.name
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, os.rename, str(file_path), str(destination_file)
                )
                logger.debug(f"Moved file: {file_path} to {destination_file}")
        except Exception as e:
            logger.error(f"Error moving file {file_path}: {e}")
            await self.move_file(file_path, ERRORS_DIR)

    async def process_file(
        self,
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
        try:
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
                await self.move_file(
                    file_path, DUPLICATES_DIR
                )  # Move file to duplicates directory
                duplicates_index[file_hash].append(
                    str(file_path)
                )  # Add file path to duplicates index
                logger.debug(f"Duplicate file found: {file_path}")
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
                await self.move_file(
                    file_path, destination_file
                )  # Move file to destination
                logger.debug(f"File processed: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            await self.move_file(file_path, ERRORS_DIR)

    async def process_directory(
        self,
        directory: Path,
        duplicates_index: Dict[str, List[str]],
        kept_files_index: Dict[str, str],
    ) -> None:
        """
        Recursively processes each file in a directory and its subdirectories.
        Employs advanced asynchronous programming techniques for efficiency.

        Args:
            directory (Path): The directory to process.
            duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
            kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
        """
        try:
            total_files = sum(1 for _ in directory.rglob("*"))
            processed_files = 0

            loop = asyncio.get_running_loop()
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.is_file():
                        file_path = Path(
                            entry.path
                        )  # Define file_path before the try block
                        try:
                            # Convert the synchronous os.DirEntry to an asynchronous Path object
                            file_hash = await self.calculate_file_hash(file_path)
                            await self.process_file(
                                file_path, file_hash, duplicates_index, kept_files_index
                            )
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
                            await self.move_file(file_path, ERRORS_DIR)
                    elif entry.is_dir():
                        # Recursively process directories
                        await self.process_directory(
                            Path(entry.path), duplicates_index, kept_files_index
                        )

            self.finished.emit()
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")

    async def run(self) -> None:
        """
        Initiates the file processing operation asynchronously.
        """
        try:
            await self.process_directory(
                self.directory, self.duplicates_index, self.kept_files_index
            )
        except Exception as e:
            logger.error(f"Error running file operations: {e}")


class FileBrowserMainWindow(QMainWindow):
    """
    The main window for the file browser application.
    Displays a tree view of the file system and a progress bar for file operations.
    """

    thread: Optional[QThread] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.setWindowTitle("File Browser")
        self.setGeometry(100, 100, 800, 600)

        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setAnimated(False)
        self.tree_view.setIndentation(20)
        self.tree_view.setSortingEnabled(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        layout = QVBoxLayout()
        widget = QWidget()
        layout.addWidget(self.tree_view)
        layout.addWidget(self.progress_bar)
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def start_file_processing(self) -> None:
        """
        Initiates the file processing operation in a separate thread.
        This method creates a new thread and worker object for processing files asynchronously,
        connecting signals and slots to update the progress bar and manage thread lifecycle.
        """
        try:
            self.thread = QThread()
            self.worker = FileOperationWorker(
                directory=Path("/path/to/directory"),
                duplicates_index={},
                kept_files_index={},
            )

            self.worker.moveToThread(self.thread)

            # Define a wrapper function to handle the coroutine
            def start_async():
                asyncio.create_task(self.worker.run())

            self.thread.started.connect(start_async)  # Use the wrapper function here
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.thread.start()
        except Exception as e:
            logger.error(f"Error starting file processing: {e}")

    def update_progress(self, value: int) -> None:
        """
        Updates the progress bar value.

        Args:
            value (int): The progress value to set.
        """
        try:
            self.progress_bar.setValue(value)
        except Exception as e:
            logger.error(f"Error updating progress bar: {e}")


def launch_file_browser(app: QApplication) -> Optional[Path]:
    """
    Launches the file browser GUI, returning the selected directory path.

    Args:
        app (QApplication): The Qt application instance.

    Returns:
        Optional[Path]: The selected directory path from the file browser GUI, or None if no selection was made.
    """
    try:
        file_browser = FileBrowserMainWindow()
        file_browser.show()
        app.exec_()
        selected_indexes = file_browser.tree_view.selectedIndexes()
        if selected_indexes:
            selected_path = Path(file_browser.model.filePath(selected_indexes[0]))
            return selected_path
        return None
    except Exception as e:
        logger.error(f"Error launching file browser: {e}")
        return None


def run_asyncio_loop(loop: asyncio.AbstractEventLoop, coro: Coroutine) -> None:
    """
    Runs the provided coroutine in the specified asyncio event loop.

    Args:
        loop (asyncio.AbstractEventLoop): The asyncio event loop to use.
        coro (Coroutine): The coroutine to run in the event loop.
    """
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Error running asyncio loop: {e}")


async def async_main(app: QApplication) -> None:
    """
    The main asynchronous function to be executed.

    Args:
        app (QApplication): The Qt application instance.
    """
    try:
        selected_directory = launch_file_browser(app)
        if selected_directory:
            duplicates_index: Dict[str, List[str]] = {}
            kept_files_index: Dict[str, str] = {}
            worker = FileOperationWorker(
                selected_directory, duplicates_index, kept_files_index
            )
            await worker.run()
        else:
            logger.info("No directory selected. Exiting.")
    except Exception as e:
        logger.error(f"Error in async main: {e}")


def is_admin() -> bool:
    """
    Determines if the script is running with administrator or superuser privileges across different operating systems,
    including Windows, Linux, and Android. It ensures that on Android, the script does not run as admin unless the user
    has superuser privileges, which is generally not common. If not running with the required privileges, it prompts the
    user to relaunch the program with elevated privileges.

    Returns:
        bool: True if the script has administrator privileges or is running as root/superuser, False otherwise.
    """
    try:
        if os.name == "nt":  # Checks if the operating system is Windows
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            if not is_admin:
                logging.info("Requesting elevated privileges on Windows.")
                response = input(
                    "This script requires administrator privileges. Would you like to relaunch with elevated privileges? (y/n): "
                )
                if response.lower() == "y":
                    ctypes.windll.shell32.ShellExecuteW(
                        None, "runas", sys.executable, " ".join(sys.argv), None, 1
                    )
                    sys.exit(0)
            return is_admin
        elif (
            os.name == "posix"
        ):  # Checks if the operating system is POSIX-compliant (Linux, Unix, macOS, etc.)
            is_root = os.geteuid() == 0
            if not is_root:
                logging.info("Requesting superuser privileges on POSIX systems.")
                response = input(
                    "This script requires superuser privileges. Would you like to attempt to relaunch with sudo? (y/n): "
                )
                if response.lower() == "y":
                    os.execvp("sudo", ["sudo"] + sys.argv)
            return is_root
        elif os.name == "android":  # Specific check for Android operating system
            # Android does not typically allow applications to run with superuser privileges unless rooted
            # This check attempts to run a command that requires superuser privileges and checks the outcome
            is_superuser = os.system("su -c id") == 0
            if not is_superuser:
                logging.info(
                    "Superuser privileges are required on Android, but not available."
                )
            return is_superuser
        else:
            # For any other operating systems not explicitly checked, direct to support for upgrade/implementation
            logging.info(
                "Operating system not currently supported. Contact support for implementation."
            )
            return False
    except Exception as e:
        # Log any exceptions that occur during the check for administrative privileges
        logging.error(f"Error checking admin privileges: {e}")
        return False


def main() -> None:
    """
    The main entry point of the script.
    """
    try:
        if is_admin():
            app = QApplication(sys.argv)
            loop = asyncio.get_event_loop()  # Get the default asyncio event loop
            loop.run_until_complete(
                async_main(app)
            )  # Run the async_main coroutine within the asyncio event loop
        else:
            logging.error("Administrator privileges required. Exiting.")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
