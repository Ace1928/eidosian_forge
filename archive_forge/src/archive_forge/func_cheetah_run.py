from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
def cheetah_run(from_pixels=True, height=64, width=64, frame_skip=2, channels_first=True):
    return DMCEnv('cheetah', 'run', from_pixels=from_pixels, height=height, width=width, frame_skip=frame_skip, channels_first=channels_first)