"""
sound_fetch.py

Author: [Author Name]
Creation Date: [Creation Date]

This module is designed to download soundfont files from a specified URL. It handles the fetching of soundfont JSON configurations, downloading instrument JSON files, and the associated pitch/velocity MP3 files for each instrument. It ensures that only missing files are downloaded to optimize bandwidth usage.

GitHub Issue Reference: https://github.com/magenta/magenta-js/issues/164
"""

import json
import os
import urllib.request
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_pitches_array(min_pitch: int, max_pitch: int) -> list:
    """
    Generates a list of pitches within the specified range.

    Parameters:
    - min_pitch (int): The minimum pitch value.
    - max_pitch (int): The maximum pitch value.

    Returns:
    - list: A list of integers representing pitches.

    Example:
    >>> get_pitches_array(60, 62)
    [60, 61, 62]
    """
    return list(range(min_pitch, max_pitch + 1))


# Constants for base URL and soundfont path
base_url = "https://storage.googleapis.com/magentadata/js/soundfonts"
soundfont_path = "sgm_plus"
soundfont_json_url = f"{base_url}/{soundfont_path}/soundfont.json"

# Attempt to download soundfont.json if it does not exist locally
soundfont_json = ""
try:
    if not os.path.exists("soundfont.json"):
        with urllib.request.urlopen(soundfont_json_url) as response:
            soundfont_json = response.read()
        with open("soundfont.json", "wb") as file:
            file.write(soundfont_json)
    else:
        with open("soundfont.json", "rb") as file:
            soundfont_json = file.read()
except Exception as e:
    logging.error(f"Failed to download or read soundfont.json due to {e}")

# Parse soundfont.json
try:
    soundfont_data = json.loads(soundfont_json)
except json.JSONDecodeError as e:
    logging.error(f"Failed to parse soundfont.json due to {e}")
    soundfont_data = None

if soundfont_data is not None:
    for instrument_id, instrument_name in soundfont_data["instruments"].items():
        if not os.path.isdir(instrument_name):
            os.makedirs(instrument_name)
        instrument_json = ""
        instrument_path = f"{soundfont_path}/{instrument_name}"
        try:
            if not os.path.exists(f"{instrument_name}/instrument.json"):
                instrument_json_url = f"{base_url}/{instrument_path}/instrument.json"
                with urllib.request.urlopen(instrument_json_url) as response:
                    instrument_json = response.read()
                with open(f"{instrument_name}/instrument.json", "wb") as file:
                    file.write(instrument_json)
            else:
                with open(f"{instrument_name}/instrument.json", "rb") as file:
                    instrument_json = file.read()
        except Exception as e:
            logging.error(
                f"Failed to download or read {instrument_name}/instrument.json due to {e}"
            )
        try:
            instrument_data = json.loads(instrument_json)
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to parse {instrument_name}/instrument.json due to {e}"
            )
            instrument_data = None
        if instrument_data is not None:
            for velocity in instrument_data["velocities"]:
                pitches = get_pitches_array(
                    instrument_data["minPitch"], instrument_data["maxPitch"]
                )
                for pitch in pitches:
                    file_name = f"p{pitch}_v{velocity}.mp3"
                    if not os.path.exists(f"{instrument_name}/{file_name}"):
                        file_url = f"{base_url}/{instrument_path}/{file_name}"
                        try:
                            with urllib.request.urlopen(file_url) as response:
                                file_contents = response.read()
                            with open(f"{instrument_name}/{file_name}", "wb") as file:
                                file.write(file_contents)
                            logging.info(f"Downloaded {instrument_name}/{file_name}")
                        except Exception as e:
                            logging.error(
                                f"Failed to download {instrument_name}/{file_name} due to {e}"
                            )
else:
    logging.error("Failed to parse soundfont.json")

"""
TODO:
- Refactor the script to encapsulate logic within functions or classes for better reusability and clarity.
- Implement a logging mechanism to replace print statements for more granular debugging and operational insight.
- Enhance error handling by specifying exception types and adding more comprehensive error messages.
- Consider adding progress indicators for file downloads to improve user experience.

Known Issues:
- Global variables are used extensively, which could lead to issues when integrating with other modules or scaling the script.
- Lack of a main guard (`if __name__ == "__main__":`) makes the script execute all operations on import, which might not be desirable.
"""
