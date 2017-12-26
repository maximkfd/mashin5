def spearman(x, y, k=None):
    correlation = {}

    rank_dict_y, ligament_y = get_rank_ligaments(y)
    for i in range(len(x)):
        correlation[i] = count_spearman_correlation_coefficient(x[i], y, rank_dict_y, ligament_y)

    correlation = sorted(correlation.items(), key=lambda d: abs(d[1]), reverse=True)

    if k is None:
        return correlation
    else:
        return correlation[0:k]


def count_spearman_correlation_coefficient(x, y, rank_dict_y, ligament_y):
    rank_dict_x, ligament_x = get_rank_ligaments(x)
    n = len(x)
    diff_rank = 0

    for i, x_el in enumerate(x):
        diff_rank += (rank_dict_x[x_el] - (n + 1) / 2) * (rank_dict_y[y[i]] - (n + 1) / 2)

    coeff = diff_rank / (n * (n - 1) * (n + 1) - (ligament_x + ligament_y))
    return coeff


def define_rank(arr):
    rank_dict = {}
    count_dict = {}

    y = sorted(arr)
    for i, y_el in enumerate(y):
        rank_dict[y_el] = i
        if y_el not in count_dict:
            count_dict[y_el] = 0

        count_dict[y_el] += 1

    for key in count_dict.keys():
        rank_dict[key] /= count_dict[key]

    return rank_dict, count_dict


def get_rank_ligaments(arr):
    rank_dict, count_dict = define_rank(arr)
    ligaments = 0
    for key in count_dict.keys():
        ligaments += count_dict[key] * ((count_dict[key] ** 2) - 1)

    ligaments *= 1 / 2
    return rank_dict, ligaments
