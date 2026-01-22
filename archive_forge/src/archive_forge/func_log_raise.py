def log_raise(self, logger, error_level=40):
    """Log problem, raise error if problem >= `error_level`

        Parameters
        ----------
        logger : log
           log object, implementing ``log`` method
        error_level : int, optional
           If ``self.problem_level`` >= `error_level`, raise error
        """
    logger.log(self.problem_level, self.message)
    if self.problem_level and self.problem_level >= error_level:
        if self.error:
            raise self.error(self.problem_msg)