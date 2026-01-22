import pathlib
import gym
from matplotlib import pyplot as plt
import minerl  # Registers envs
def psuedo_main():
    for env_name in ENV_NAMES:
        output_dir = OUTPUT_ROOT / env_name
        for seed in range(N_SEEDS):
            for trie in range(N_TRYS):
                with gym.make(env_name) as env:
                    env.seed(seed)
                    obs = env.reset()
                    for n_steps in range(max(STEP_RANGES) + 1):
                        if n_steps in STEP_RANGES:
                            save_path = output_dir / f'seed{seed:02d}_{n_steps:02d}_{trie:02d}.png'
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            plt.imsave(save_path, obs['pov'])
                            print(f'Saved a screenshot to {str(save_path)}')