import deap
from copy import copy
def pipeline_code_wrapper(pipeline_code, random_state=None):
    """Generate code specific to the execution of the sklearn pipeline.

    Parameters
    ----------
    pipeline_code: str
        Code that defines the final sklearn pipeline
    random_state: integer or None
        Random seed in train_test_split function and exported pipeline.

    Returns
    -------
    exported_code: str
        Source code for the sklearn pipeline and calls to fit and predict

    """
    if random_state is None:
        exported_code = 'exported_pipeline = {}\n\nexported_pipeline.fit(training_features, training_target)\nresults = exported_pipeline.predict(testing_features)\n'.format(pipeline_code)
    elif pipeline_code.startswith('make_pipeline'):
        exported_code = "exported_pipeline = {}\n# Fix random state for all the steps in exported pipeline\nset_param_recursive(exported_pipeline.steps, 'random_state', {})\n\nexported_pipeline.fit(training_features, training_target)\nresults = exported_pipeline.predict(testing_features)\n".format(pipeline_code, random_state)
    else:
        exported_code = "exported_pipeline = {}\n# Fix random state in exported estimator\nif hasattr(exported_pipeline, 'random_state'):\n    setattr(exported_pipeline, 'random_state', {})\n\nexported_pipeline.fit(training_features, training_target)\nresults = exported_pipeline.predict(testing_features)\n".format(pipeline_code, random_state)
    return exported_code