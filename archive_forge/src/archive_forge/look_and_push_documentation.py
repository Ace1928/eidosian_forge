import gymnasium as gym
import numpy as np
Memory-requiring Env: Best sequence of actions depends on prev. states.

    Optimal behavior:
        0) a=0 -> observe next state (s'), which is the "hidden" state.
            If a=1 here, the hidden state is not observed.
        1) a=1 to always jump to s=2 (not matter what the prev. state was).
        2) a=1 to move to s=3.
        3) a=1 to move to s=4.
        4) a=0 OR 1 depending on s' observed after 0): +10 reward and done.
            otherwise: -10 reward and done.
    