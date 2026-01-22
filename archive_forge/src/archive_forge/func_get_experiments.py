from benchmark.experiments import structured_data
def get_experiments(task_name):
    if task_name == 'structured_data_classification':
        return [structured_data.Titanic(), structured_data.Iris(), structured_data.Wine()]