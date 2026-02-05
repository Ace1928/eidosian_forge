import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"pkg_resources",
)
warnings.filterwarnings(
    "ignore",
    message="Deprecated call to `pkg_resources.declare_namespace",
    category=DeprecationWarning,
)

repo_root = Path(__file__).resolve().parents[2]
game_src = repo_root / "game_forge" / "src"
lib_dir = repo_root / "lib"
for path in (repo_root, game_src, lib_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
