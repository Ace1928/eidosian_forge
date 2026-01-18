#!/usr/bin/env python3

"""
Script to download and install standard FIGlet fonts for Figlet Forge.

This script fetches the standard FIGlet fonts from the official repository
and installs them into the Figlet Forge package for use.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

# URLs for font sources
FIGLET_FONTS_URL = "http://ftp.figlet.org/pub/figlet/fonts/contributed.tar.gz"
STANDARD_FONTS_URL = "http://ftp.figlet.org/pub/figlet/fonts/ours.tar.gz"
GITHUB_FONTS_URL = "https://github.com/xero/figlet-fonts/archive/master.zip"

# Package path
package_root = Path(__file__).parent
fonts_dir = package_root / "src" / "figlet_forge" / "fonts"


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, dest):
    """Download a file from URL to destination."""
    print(f"Downloading {url}...")
    try:
        urlretrieve(url, dest)
        print(f"Downloaded to {dest}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_tar_gz(file_path, extract_dir):
    """Extract .tar.gz file."""
    print(f"Extracting {file_path}...")
    try:
        if sys.platform == "win32":
            # Use Python's subprocess on Windows
            subprocess.run(["tar", "-xzf", file_path, "-C", extract_dir], check=True)
        else:
            # Use tar command on Unix-like systems
            subprocess.run(["tar", "-xzf", file_path, "-C", extract_dir], check=True)
        print(f"Extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return False


def extract_zip(file_path, extract_dir):
    """Extract .zip file."""
    print(f"Extracting {file_path}...")
    try:
        with ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return False


def copy_fonts(source_dir, dest_dir):
    """Copy .flf font files to destination."""
    ensure_dir(dest_dir)
    count = 0

    # Walk through all subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".flf"):
                src_file = Path(root) / file
                dest_file = dest_dir / file
                try:
                    shutil.copy2(src_file, dest_file)
                    count += 1
                except Exception as e:
                    print(f"Error copying {src_file}: {e}")

    print(f"Copied {count} font files to {dest_dir}")
    return count


def main():
    """Main function to download and install fonts."""
    print("Figlet Forge Font Installer")
    print("===========================")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Using temporary directory: {temp_path}")

        # Download and extract standard fonts
        std_fonts_file = temp_path / "ours.tar.gz"
        if download_file(STANDARD_FONTS_URL, std_fonts_file):
            extract_tar_gz(std_fonts_file, temp_dir)

        # Download and extract contributed fonts
        contrib_fonts_file = temp_path / "contributed.tar.gz"
        if download_file(FIGLET_FONTS_URL, contrib_fonts_file):
            extract_tar_gz(contrib_fonts_file, temp_dir)

        # Download and extract GitHub collection
        github_fonts_file = temp_path / "github_fonts.zip"
        if download_file(GITHUB_FONTS_URL, github_fonts_file):
            extract_zip(github_fonts_file, temp_dir)

        # Copy fonts to package
        print("\nCopying fonts to package...")
        ensure_dir(fonts_dir)
        font_count = copy_fonts(temp_path, fonts_dir)

        if font_count > 0:
            print(f"\n✅ Successfully installed {font_count} fonts")
        else:
            print("\n❌ No fonts were installed")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
