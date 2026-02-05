# -------------------------------------------------------------------------------------
# ENHANCED PROFILE MANAGEMENT - Encapsulated within UserManager Class
# -------------------------------------------------------------------------------------

import os  # For file system operations, like checking if a file exists and replacing files.
import json  # For encoding and decoding player profile data to and from JSON format.
import logging  # For structured logging of events, errors, and debugging information.
from typing import (
    Dict,
    Any,
)  # For type hinting, improving code readability and maintainability.
from eidosian_core import eidosian


DEFAULT_PROFILE = {  # Define a default profile here for testing purposes if init.py is not accessible in test context
    "setting1": "value1",
    "setting2": 100,
}
PLAYER_PROFILES_JSON = "test_profiles.json"  # Define a test profile json path here for testing purposes if init.py is not accessible in test context


class UserManager:
    """
    Manages player profile data, including loading from and saving to a JSON file.

    This class encapsulates the functionality for handling player profiles, providing
    methods to load profiles from a JSON file, save profiles to a JSON file, and
    ensure data integrity through validation and atomic write operations.

    It is designed to be robust, providing detailed logging for debugging and operational
    insights, and handling potential errors gracefully to prevent data loss or corruption.
    """

    def __init__(
        self, profiles_file_path: str, default_profile_template: Dict[str, Any]
    ):
        """
        Initializes the UserManager with the path to the profiles JSON file and a default profile template.

        Args:
            profiles_file_path (str): The file path to the JSON file where player profiles are stored.
            default_profile_template (Dict[str, Any]): A dictionary defining the default structure of a player profile.
                                                     This template is used to ensure all profiles have a consistent format
                                                     and to populate missing fields when loading profiles.
        """
        self.profiles_file_path = profiles_file_path
        self.default_profile_template = default_profile_template
        logging.debug(
            f"UserManager initialized with profiles file: {self.profiles_file_path}"
        )

    @eidosian()
    def load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Load player profiles from a JSON file, with enhanced validation and default structure.

        This method is responsible for loading player profiles from the JSON file specified in `self.profiles_file_path`.
        It performs several key operations to ensure data integrity and consistency:

        1. **File Existence Check**: Verifies if the profiles file exists at the given path. If not, it logs a warning and
           returns an empty dictionary, preventing errors and starting with a clean state if no profiles are found.

        2. **JSON Loading and Parsing**: Attempts to open and read the JSON file. It uses `json.load()` to parse the JSON
           content into a Python dictionary. If the file is corrupted or not in valid JSON format, it catches the
           `json.JSONDecodeError`, logs an error, and returns an empty dictionary to avoid program crashes.

        3. **Profile Validation and Default Application**: After successfully loading the JSON data, it iterates through each
           player profile. For each profile, it checks for the presence of all keys defined in the `self.default_profile_template`.
           If any key is missing in a profile, it populates that key with the corresponding default value from the template.
           This ensures that all profiles adhere to a consistent structure and prevents issues arising from missing data fields.

        4. **Error Handling**: Implements comprehensive error handling using try-except blocks to gracefully manage potential
           issues such as `FileNotFoundError`, `json.JSONDecodeError`, and other unexpected exceptions during file operations.
           In case of any error, it logs the error details and returns an empty dictionary, ensuring the application does not crash
           and can continue operation, albeit without loaded profiles.

        5. **Logging**: Utilizes the `logging` module to provide detailed logs at different levels (INFO, DEBUG, WARNING, ERROR).
           This helps in monitoring the profile loading process, debugging potential issues, and providing operational insights.
           Logs include information about file existence, successful loading, data validation, and any errors encountered.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing player profiles, where keys are player names and values
                                     are their profile dictionaries. Returns an empty dictionary if the profile file is not
                                     found, if there is an error in JSON decoding, or any other exception during the loading process.
        """
        logging.info("Loading player profiles...")
        # Check if the player profiles JSON file exists at the specified path.
        if not os.path.exists(self.profiles_file_path):
            logging.warning(
                f"Player profiles file not found: {self.profiles_file_path}. Returning empty profiles."
            )
            return {}  # Return an empty dictionary if the file does not exist.

        # Attempt to read and parse the JSON file.
        try:
            with open(self.profiles_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # Load JSON data from the file.
                logging.debug(
                    f"Successfully loaded player profiles from {self.profiles_file_path}: {data}"
                )

                # Validate and update existing profiles with any missing fields using the default profile template.
                for player_name, profile in data.items():
                    for key, default_value in self.default_profile_template.items():
                        if key not in profile:
                            profile[key] = (
                                default_value  # Assign default value if key is missing.
                            )
                            logging.debug(
                                f"Player {player_name}: Added missing key '{key}' with default value '{default_value}'."
                            )
                logging.info("Player profiles loaded and validated successfully.")
                return data  # Return the loaded and validated player profiles.

        except FileNotFoundError:
            # This exception is technically redundant because of the initial os.path.exists check,
            # but it's included for robustness in case of race conditions or unexpected file system behavior.
            logging.warning(
                f"Profile file was not found (FileNotFoundError): {self.profiles_file_path}. Returning empty profiles."
            )
            return {}  # Return empty profiles if file not found during file opening.
        except json.JSONDecodeError as e:
            # Handle JSON decoding errors, which occur if the file content is not valid JSON.
            logging.error(
                f"Error decoding JSON from file {self.profiles_file_path}: {e}. The file may be corrupted or not a valid JSON file. Returning empty profiles."
            )
            return {}  # Return empty profiles if JSON decoding fails.
        except Exception as e:
            # Catch-all for any other unexpected exceptions during file loading or processing.
            logging.error(
                f"Unexpected error loading player profiles from {self.profiles_file_path}: {e}. Returning empty profiles."
            )
            return {}  # Return empty profiles in case of any unhandled exception.

    @eidosian()
    def save_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> None:
        """
        Save player profiles to a JSON file using an atomic write pattern to prevent data corruption.

        This method saves the provided player profiles to the JSON file specified by `self.profiles_file_path`.
        It employs an atomic write strategy to ensure data integrity, especially in scenarios where the saving
        process might be interrupted (e.g., system crash, power failure). The atomic write process involves:

        1. **Writing to a Temporary File**: First, the method writes the profile data to a temporary file (`.tmp` extension)
           in the same directory as the target profiles file. This ensures that if any error occurs during the writing process,
           the original profiles file remains untouched and in a consistent state.

        2. **Atomic Replacement**: Once the data is successfully written to the temporary file, the method uses `os.replace()`
           to atomically replace the original profiles file with the temporary file. The `os.replace()` operation is crucial
           because it is typically an atomic operation on most operating systems. This means that the replacement is done
           as a single, uninterruptible step, preventing data corruption or loss that could occur if the process was interrupted
           between deleting the old file and writing the new one.

        3. **Error Handling**: The method includes robust error handling using a try-except block to catch any exceptions
           that might occur during file operations (e.g., disk errors, permission issues). If an error is caught, it logs
           the error details, including the exception message and the file path, ensuring that any saving failures are
           properly recorded and can be diagnosed. In case of an error, it also logs a critical error message indicating
           that the profile saving operation has failed and data might be lost or corrupted, prompting further investigation
           or user notification in a real-world application.

        4. **Logging**: Comprehensive logging is implemented at different levels (INFO, DEBUG, ERROR) to track the saving process.
           It logs the start of the saving operation, successful writing to the temporary file, successful replacement of the
           original file, and any errors encountered. Debug logs provide detailed information about the file operations,
           while info and error logs provide higher-level status updates and error notifications.

        Args:
            profiles (Dict[str, Dict[str, Any]]): A dictionary containing player profiles to save.
                                                 Keys are player names, and values are their profile dictionaries.
        """
        logging.info("Saving player profiles...")
        temp_file_path = (
            self.profiles_file_path + ".tmp"
        )  # Construct the path for the temporary file.

        try:
            # Write player profiles to a temporary file first.
            with open(temp_file_path, "w", encoding="utf-8") as f:
                json.dump(
                    profiles, f, indent=2
                )  # Serialize the profiles dictionary to JSON and write to the temporary file with indentation for readability.
                logging.debug(
                    f"Successfully wrote player profiles to temporary file: {temp_file_path}"
                )

            # Atomically replace the original file with the temporary file.
            os.replace(temp_file_path, self.profiles_file_path)
            logging.debug(
                f"Successfully replaced {self.profiles_file_path} with {temp_file_path}"
            )
            logging.info("Player profiles saved successfully.")

        except Exception as e:
            # Handle any exceptions that occur during the file saving process.
            logging.error(
                f"Error saving player profiles to {self.profiles_file_path}: {e}"
            )
            logging.error(
                "Profile saving operation failed. Data might not be saved or might be corrupted."
            )
            # In a production environment, consider more robust error handling, such as user notifications or retry mechanisms.
            # It's crucial to inform the user if profile saving fails to prevent data loss.


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)

    TEST_PROFILES_FILE = "test_user_profiles.json"
    TEST_DEFAULT_PROFILE = {"test_setting": "default_value", "score": 0}
    test_user_manager = UserManager(TEST_PROFILES_FILE, TEST_DEFAULT_PROFILE)

    print("Running UserManager tests...")

    # --- Test 1: Load profiles - file does not exist ---
    print("\nTest 1: Load profiles - file does not exist")
    if os.path.exists(TEST_PROFILES_FILE):
        os.remove(TEST_PROFILES_FILE)  # Ensure file does not exist
    profiles = test_user_manager.load_profiles()
    assert profiles == {}, "Test 1 Failed: Should return empty profiles"
    print("Test 1 Passed")

    # --- Test 2: Load profiles - valid JSON file ---
    print("\nTest 2: Load profiles - valid JSON file")
    valid_profiles = {
        "player1": {"test_setting": "player1_value", "score": 10},
        "player2": {"test_setting": "player2_value", "score": 20},
    }
    with open(TEST_PROFILES_FILE, "w") as f:
        json.dump(valid_profiles, f)
    loaded_profiles = test_user_manager.load_profiles()
    assert (
        loaded_profiles == valid_profiles
    ), "Test 2 Failed: Loaded profiles do not match"
    print("Test 2 Passed")

    # --- Test 3: Load profiles - invalid JSON file ---
    print("\nTest 3: Load profiles - invalid JSON file")
    with open(TEST_PROFILES_FILE, "w") as f:
        f.write("invalid json content")
    profiles = test_user_manager.load_profiles()
    assert (
        profiles == {}
    ), "Test 3 Failed: Should return empty profiles for invalid JSON"
    print("Test 3 Passed")

    # --- Test 4: Load profiles - file exists but is empty ---
    print("\nTest 4: Load profiles - empty file")
    open(TEST_PROFILES_FILE, "w").close()  # Create an empty file
    profiles = test_user_manager.load_profiles()
    assert profiles == {}, "Test 4 Failed: Should return empty profiles for empty file"
    print("Test 4 Passed")

    # --- Test 5: Load profiles - missing keys, defaults applied ---
    print("\nTest 5: Load profiles - missing keys, defaults applied")
    missing_key_profiles = {
        "player3": {"score": 30},  # Missing 'test_setting'
        "player4": {"test_setting": "player4_value"},  # Missing 'score'
    }
    expected_profiles_with_defaults = {
        "player3": {"score": 30, "test_setting": "default_value"},
        "player4": {"test_setting": "player4_value", "score": 0},
    }

    with open(TEST_PROFILES_FILE, "w") as f:
        json.dump(missing_key_profiles, f)
    loaded_profiles_with_defaults = test_user_manager.load_profiles()

    assert (
        loaded_profiles_with_defaults == expected_profiles_with_defaults
    ), "Test 5 Failed: Defaults not applied correctly"
    print("Test 5 Passed")

    # --- Test 6: Save profiles - valid profiles ---
    print("\nTest 6: Save profiles - valid profiles")
    profiles_to_save = {
        "player5": {"test_setting": "player5_value", "score": 50},
        "player6": {"test_setting": "player6_value", "score": 60},
    }
    test_user_manager.save_profiles(profiles_to_save)
    loaded_after_save = test_user_manager.load_profiles()  # Load back to verify save
    assert (
        loaded_after_save == profiles_to_save
    ), "Test 6 Failed: Saved profiles not loaded correctly"
    print("Test 6 Passed")

    # --- Cleanup: Remove test file ---
    os.remove(TEST_PROFILES_FILE)
    print("\nAll UserManager tests completed successfully.")
