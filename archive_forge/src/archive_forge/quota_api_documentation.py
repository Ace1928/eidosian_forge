import abc
Return the quota driver workers to be spawned during initialization

        This method returns the quota driver workers that needs to be spawned
        during the plugin initialization. For example, ``DbQuotaNoLockDriver``
        requires a ``PeriodicWorker`` to clean up the expired reservations left
        in the database.

        :return: list of ``worker.BaseWorker`` or derived instances.
        