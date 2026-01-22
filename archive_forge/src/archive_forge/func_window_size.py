from aiokafka.metrics.measurable_stat import AbstractMeasurableStat
from aiokafka.metrics.stats.sampled_stat import AbstractSampledStat
def window_size(self, config, now):
    self._stat.purge_obsolete_samples(config, now)
    '\n        Here we check the total amount of time elapsed since the oldest\n        non-obsolete window. This give the total window_size of the batch\n        which is the time used for Rate computation. However, there is\n        an issue if we do not have sufficient data for e.g. if only\n        1 second has elapsed in a 30 second window, the measured rate\n        will be very high. Hence we assume that the elapsed time is\n        always N-1 complete windows plus whatever fraction of the final\n        window is complete.\n\n        Note that we could simply count the amount of time elapsed in\n        the current window and add n-1 windows to get the total time,\n        but this approach does not account for sleeps. AbstractSampledStat\n        only creates samples whenever record is called, if no record is\n        called for a period of time that time is not accounted for in\n        window_size and produces incorrect results.\n        '
    total_elapsed_time_ms = now - self._stat.oldest(now).last_window_ms
    num_full_windows = int(total_elapsed_time_ms / config.time_window_ms)
    min_full_windows = config.samples - 1
    if num_full_windows < min_full_windows:
        total_elapsed_time_ms += (min_full_windows - num_full_windows) * config.time_window_ms
    return total_elapsed_time_ms