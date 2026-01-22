import collections
from rich.console import Console
from rich.table import Table
import typer
from ray.rllib import train as train_module
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import (
@app.callback()
def main_helper():
    """Welcome to the

    .                                                  ╔▄▓▓▓▓▄

    .                                                ╔██▀╙╙╙▀██▄

    . ╫█████████████▓   ╫████▓             ╫████▓    ██▌     ▐██   ╫████▒

    . ╫███████████████▓ ╫█████▓            ╫█████▓   ╫██     ╫██   ╫██████▒

    . ╫█████▓     ████▓ ╫█████▓            ╫█████▓    ╙▓██████▀    ╫██████████████▒

    . ╫███████████████▓ ╫█████▓            ╫█████▓       ╫█▒       ╫████████████████▒

    . ╫█████████████▓   ╫█████▓            ╫█████▓       ╫█▒       ╫██████▒    ╫█████▒

    . ╫█████▓███████▓   ╫█████▓            ╫█████▓       ╫█▒       ╫██████▒    ╫█████▒

    . ╫█████▓   ██████▓ ╫████████████████▄ ╫█████▓       ╫█▒       ╫████████████████▒

    . ╫█████▓     ████▓ ╫█████████████████ ╫█████▓       ╫█▒       ╫██████████████▒

    .                                        ╣▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▄

    .                                        ╫██╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╙╫█▒

    .                                        ╫█  Command Line Interface █▒

    .                                        ╫██▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄╣█▒

    .                                         ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

    .

        Example usage for training:

            rllib train --algo DQN --env CartPole-v1

            rllib train file tuned_examples/ppo/pendulum-ppo.yaml



        Example usage for evaluation:

            rllib evaluate /trial_dir/checkpoint_000001/checkpoint-1 --algo DQN



        Example usage for built-in examples:

            rllib example list

            rllib example get atari-ppo

            rllib example run atari-ppo

    """