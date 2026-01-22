import yaml
import os
def extract_matrix(workflow_yaml):
    jobs = extract_value('jobs', workflow_yaml)
    integration = extract_value('integration', jobs)
    strategy = extract_value('strategy', integration)
    matrix = extract_value('matrix', strategy)
    return matrix