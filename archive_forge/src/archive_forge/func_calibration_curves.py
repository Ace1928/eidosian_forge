from warnings import simplefilter
import numpy as np
import sklearn
from sklearn import model_selection, naive_bayes
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import wandb
from wandb.sklearn import utils
def calibration_curves(clf, X, y, clf_name):
    if int(sklearn.__version__.split('.')[1]) >= 20 and isinstance(clf, naive_bayes.ComplementNB):
        X = X - X.min()
    isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
    lr = LogisticRegression(C=1.0)
    model_column = []
    frac_positives_column = []
    mean_pred_value_column = []
    hist_column = []
    edge_column = []
    model_column.append('Perfectly calibrated')
    frac_positives_column.append(0)
    mean_pred_value_column.append(0)
    hist_column.append(0)
    edge_column.append(0)
    model_column.append('Perfectly calibrated')
    hist_column.append(0)
    edge_column.append(0)
    frac_positives_column.append(1)
    mean_pred_value_column.append(1)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.9, random_state=42)
    models = [lr, isotonic, sigmoid]
    names = ['Logistic', f'{clf_name} Isotonic', f'{clf_name} Sigmoid']
    for model, name in zip(models, names):
        model.fit(X_train, y_train)
        if hasattr(model, 'predict_proba'):
            prob_pos = model.predict_proba(X_test)[:, 1]
        else:
            prob_pos = model.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        hist, edges = np.histogram(prob_pos, bins=10, density=False)
        frac_positives, mean_pred_value = sklearn.calibration.calibration_curve(y_test, prob_pos, n_bins=10)
        num_entries = len(frac_positives)
        for i in range(num_entries):
            hist_column.append(hist[i])
            edge_column.append(edges[i])
            model_column.append(name)
            frac_positives_column.append(utils.round_3(frac_positives[i]))
            mean_pred_value_column.append(utils.round_3(mean_pred_value[i]))
            if utils.check_against_limit(i, 'calibration_curve', utils.chart_limit - 2):
                break
    table = make_table(model_column, frac_positives_column, mean_pred_value_column, hist_column, edge_column)
    chart = wandb.visualize('wandb/calibration/v1', table)
    return chart