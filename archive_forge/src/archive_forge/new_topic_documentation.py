from aiokafka.errors import IllegalArgumentError
 A class for new topic creation
    Arguments:
        name (string): name of the topic
        num_partitions (int): number of partitions
            or -1 if replica_assignment has been specified
        replication_factor (int): replication factor or -1 if
            replica assignment is specified
        replica_assignment (dict of int: [int]): A mapping containing
            partition id and replicas to assign to it.
        topic_configs (dict of str: str): A mapping of config key
            and value for the topic.
    