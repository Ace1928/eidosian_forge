from unittest import mock
import keras_tuner
from tensorflow import keras
import autokeras as ak
from autokeras import test_utils
from autokeras.tuners import greedy
from autokeras.tuners import task_specific
@mock.patch('autokeras.tuners.greedy.GreedyOracle.get_best_trials')
def test_greedy_oracle_populate_doesnt_crash_with_init_hps(get_best_trials):
    hp = keras_tuner.HyperParameters()
    keras.backend.clear_session()
    input_node = ak.ImageInput(shape=(32, 32, 3))
    input_node.batch_size = 32
    input_node.num_samples = 1000
    output_node = ak.ImageBlock()(input_node)
    head = ak.ClassificationHead(num_classes=10)
    head.shape = (10,)
    output_node = head(output_node)
    graph = ak.graph.Graph(inputs=input_node, outputs=output_node)
    graph.build(hp)
    oracle = greedy.GreedyOracle(initial_hps=task_specific.IMAGE_CLASSIFIER, objective='val_loss', seed=test_utils.SEED)
    trial = mock.Mock()
    trial.hyperparameters = hp
    get_best_trials.return_value = [trial]
    for i in range(10):
        keras.backend.clear_session()
        values = oracle.populate_space('a')['values']
        hp = oracle.hyperparameters.copy()
        hp.values = values
        graph.build(hp)
        oracle.update_space(hp)