from sklearn import svm
from fs import *


def check_svm(features_train, labels_train, features_test, labels_test, mask):
    features_test = squeze(features_test, mask)
    features_train = squeze(features_train, mask)
    labels_train = (labels_train + 1) // 2
    labels_test = (labels_test + 1) // 2
    clf = svm.SVC()
    clf.fit(features_train, labels_train)

    tp = 0.001
    tn = 0
    fn = 0
    fp = 0
    predictions = clf.predict(features_test)
    for prediction, label in zip(predictions, labels_test):
        if prediction == label:
            if prediction == 1:
                tp += 1
            else:
                tn += 1
        else:
            if prediction == 1:
                fp += 1
            else:
                fn += 1
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f = 2 * precision * recall / (precision + recall)

    print(f, (tp + tn)/len(predictions), "ones: ", len(np.where(predictions > 0)[0]), sep=" ")


def squeze(features_test, mask):
    return features_test[:, mask]
    # return np.squeeze(features_test[:, mask], axis=1)

