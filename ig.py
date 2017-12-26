import numpy as np
from sklearn.feature_selection import chi2

def information_gain(x, y):
    def _entropy(values):
        counts = np.bincount(values)
        probs = counts[np.nonzero(counts)] / float(len(values))
        return - np.sum(probs * np.log(probs))

    def _information_gain(feature, y):
        feature_set_indices = np.nonzero(feature)[0]
        feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices]
        entropy_x_set = _entropy(y[feature_set_indices])
        entropy_x_not_set = _entropy(y[feature_not_set_indices])

        return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
                                 + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))

    y = (y + 1) // 2
    feature_size = x.shape[0]
    feature_range = range(0, feature_size)
    entropy_before = _entropy(y)
    information_gain_scores = []

    for feature in x.T:
        information_gain_scores.append(_information_gain(feature, y))
    return information_gain_scores, []
    # def _calIg():
    #     entropy_x_set = 0
    #     entropy_x_not_set = 0
    #     for c in classCnt:
    #         probs = classCnt[c] / float(featureTot)
    #         entropy_x_set = entropy_x_set - probs * np.log(probs)
    #         probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
    #         entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
    #     for c in classTotCnt:
    #         if c not in classCnt:
    #             probs = classTotCnt[c] / float(tot - featureTot)
    #             entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
    #     return entropy_before - ((featureTot / float(tot)) * entropy_x_set
    #                          +  ((tot - featureTot) / float(tot)) * entropy_x_not_set)
    #
    # tot = x.shape[0]
    # y = (y + 1) // 2
    # classTotCnt = {}
    # entropy_before = 0
    # for i in y:
    #     if i not in classTotCnt:
    #         classTotCnt[i] = 1
    #     else:
    #         classTotCnt[i] = classTotCnt[i] + 1
    # for c in classTotCnt:
    #     probs = classTotCnt[c] / float(tot)
    #     entropy_before = entropy_before - probs * np.log(probs)
    #
    # nz = x.T.nonzero()
    # pre = 0
    # classCnt = {}
    # featureTot = 0
    # information_gain = []
    # for i in range(0, len(nz[0])):
    #     if (i != 0 and nz[0][i] != pre):
    #         for notappear in range(pre+1, nz[0][i]):
    #             information_gain.append(0)
    #         ig = _calIg()
    #         information_gain.append(ig)
    #         pre = nz[0][i]
    #         classCnt = {}
    #         featureTot = 0
    #     featureTot = featureTot + 1
    #     yclass = y[nz[1][i]]
    #     if yclass not in classCnt:
    #         classCnt[yclass] = 1
    #     else:
    #         classCnt[yclass] = classCnt[yclass] + 1
    # ig = _calIg()
    # information_gain.append(ig)
    #
    # return np.asarray(information_gain)