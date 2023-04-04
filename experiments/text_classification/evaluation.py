from sklearn.metrics import f1_score, recall_score, precision_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def sentence_label_evaluation(predictions, real_values):
    return macro_f1(real_values, predictions), weighted_f1(real_values, predictions)
