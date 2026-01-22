from typing import TYPE_CHECKING, Dict, Optional
Container for urls used in the wandb package.

Use this anytime a URL is displayed to the user.

Usage:
    ```python
    from wandb.sdk.lib.wburls import wburls

    print(f"This is a url {wburls.get('cli_launch')}")
    ```
