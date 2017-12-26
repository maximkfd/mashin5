from matplotlib import pyplot as plt
from matplotlib_venn import venn3

import svmy
from chi import chi2
from ig import *
from spierman import spearman


# from sklearn.feature_selection import chi2


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_data(filename='arcene_train'):
    data_filename = filename + '.data'
    labels_filename = filename + '.labels'

    labels = []
    features = []

    with open(labels_filename) as f:
        for line in f:
            labels.append(int(line))

    with open(data_filename) as f:
        for line in f:
            tmp = []
            for symbol in line.split(" "):
                try:
                    tmp.append(int(symbol))
                except ValueError:
                    pass
            features.append(tmp)
    if features[0] == []:
        features.__delitem__(0)
    return np.asarray(features), np.asarray(labels, dtype='int32')


def get_ig_indices(features, labels, k=10):
    gain = information_gain(features, labels)
    ans = np.array(gain[0]).argsort()[-k:][::-1]
    return ans


def get_chi_indices(x, y, k=20):
    all_chi = np.nan_to_num(chi2(x, y)[0])
    ans = all_chi.argsort()[-k:][::-1]
    return ans


def get_spierman_indices(x, y, k=20):
    keys = []
    gain = spearman(x.T, y, k)
    for k, v in gain:
        keys.append(k)
    return keys


if __name__ == '__main__':
    features_train, labels_train = read_data('arcene_train')
    features_test, labels_test = read_data('arcene_valid')
    chi_i = 3
    ig_i = 25
    spir_i = 100
    chi = (get_chi_indices(features_train, labels_train, k=chi_i))
    svmy.check_svm(features_train, labels_train, features_test, labels_test, chi)
    ig = get_ig_indices(features_train, labels_train, ig_i)
    svmy.check_svm(features_train, labels_train, features_test, labels_test, ig)
    spir = get_spierman_indices(features_train, labels_train, spir_i)
    svmy.check_svm(features_train, labels_train, features_test, labels_test, spir)
    # s = (
    #     len(chi),  # Pearson
    #     len(ig),  # Spearman
    #     len(np.intersect1d(chi, ig)),  # Pearson+IG
    #     len(spir),  # IG
    #     len(np.intersect1d(chi, spir)),  # Pearson+Spearman
    #     len(np.intersect1d(spir, ig)),  # Spearman+IG
    #     len(np.intersect1d(np.intersect1d(chi, spir), ig)),  # Pearson+Spearman+IG
    # )
    #
    # v = venn3(subsets=s, set_labels=('Pearson', 'Spearman', 'IG'))
    # plt.show()



