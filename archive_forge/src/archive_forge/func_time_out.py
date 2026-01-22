@property
def time_out(self):
    """
        bool : True if time out found for local or global
        separation loop, False otherwise.
        """
    local_time_out = self.solved_locally and self.local_separation_loop_results.time_out
    global_time_out = self.solved_globally and self.global_separation_loop_results.time_out
    return local_time_out or global_time_out