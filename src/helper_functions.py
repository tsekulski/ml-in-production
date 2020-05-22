### Define some helper functions
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


def predict(model, X):
    predictions = model.predict_proba(X)[:, 1]

    return predictions.tolist()


def validate(model, X, y):
    # predict
    predictions_prob = predict(model, X)
    predictions_rounded = np.round(predictions_prob)

    # find roc curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(y, predictions_prob)
    roc_curve = {'false_positive_rate': fpr.tolist(), 'true_positive_rate': tpr.tolist()}

    # find auc
    auc = sklearn.metrics.auc(fpr, tpr)

    # find accuracy
    acc = sklearn.metrics.accuracy_score(y, predictions_rounded, normalize=True)

    # find log_loss
    log_loss = sklearn.metrics.log_loss(y, predictions_prob)

    # find precision
    precision = sklearn.metrics.precision_score(y, predictions_rounded)

    # find recall
    recall = sklearn.metrics.recall_score(y, predictions_rounded)

    # find specificity
    specificity = sklearn.metrics.recall_score(1 - y, 1 - predictions_rounded)

    # find f1-score
    f1score = sklearn.metrics.f1_score(y, predictions_rounded)

    # find precision-recall curve
    prec, rec, _ = precision_recall_curve = sklearn.metrics.precision_recall_curve(y, predictions_prob)
    precision_recall_curve = {'recall': rec.tolist(), 'precision': prec.tolist()}

    # find confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(y, predictions_rounded).tolist()

    print('area_under_curve: ' + '{:.3f}'.format(auc))
    print('accuracy: ' + '{:.3f}'.format(acc))
    print('log_loss: ' + '{:.3f}'.format(log_loss))
    print('precision: ' + '{:.3f}'.format(precision))
    print('recall: ' + '{:.3f}'.format(recall))
    print('specificity: ' + '{:.3f}'.format(specificity))
    print('f1_score: ' + '{:.3f}'.format(f1score))
    print()

    print('confusion_matrix:')
    print(confusion_matrix)
    print()

    print('roc_curve:')
    plot_ROC_curve(roc_curve, auc)
    print()

    #print('precision_recall_curve:')
    #plot_precision_recall_curve(precision_recall_curve)


def plot_ROC_curve(roc_curve, auc):
    plt.figure()
    lw = 2
    plt.plot(roc_curve['false_positive_rate'], roc_curve['true_positive_rate'], color='darkorange',
             lw=lw, label='ROC curve (AUC = ' + '{:.3f}'.format(auc) + ')')
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.clf()


def plot_precision_recall_curve(precision_recall_curve):
    plt.figure()
    lw = 2
    plt.plot(precision_recall_curve['recall'], precision_recall_curve['precision'], color='blue',
             lw=lw, label='precision-recall curve')
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.clf()