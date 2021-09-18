import thinkstats2


def PmfMean(pmf):
    mean = 0
    for value, prob in pmf.Items():
        mean += value * prob
    return mean


def PmfVar(pmf):
    var = 0
    mean = PmfMean(pmf)
    for value, prob in pmf.Items():
        var += prob * (value - mean) ** 2
    return var


if __name__ == "__main__":
    print("Test.py")

    d = {7: 8, 12: 8, 17: 14, 22: 4,
         27: 6, 32: 12, 37: 8, 42: 3, 47: 2}

    pmf = thinkstats2.Pmf(d, label='actual')
    print('mean', pmf.Mean() == PmfMean(pmf), pmf.Mean(), PmfMean(pmf))
    print('var', pmf.Var() == PmfVar(pmf), pmf.Var(), PmfVar(pmf))
