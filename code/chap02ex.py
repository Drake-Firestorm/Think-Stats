"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import sys
from operator import itemgetter

import first
import thinkstats2


def Mode(hist):
    """Returns the value with the highest frequency.

    hist: Hist object

    returns: value from Hist
    """
    hist = sorted(hist, key=hist.d.get, reverse=True)
    return hist[0]


def AllModes(hist):
    """Returns value-freq pairs in decreasing order of frequency.

    hist: Hist object

    returns: iterator of value-freq pairs
    """
    li = list()
    for key in sorted(hist, key=hist.d.get, reverse=True):
        li.append((key, hist.d.get(key)))
    return li


def investigate_weight(live, firsts, others):
    """
    Investigate whether first babies are lighter or heavier than others.

    :param live: DataFrame of all live births
    :param firsts: DataFrame of all first births
    :param others: DataFrame of others
    :return: None
    """
    first_wt = firsts.totalwgt_lb
    other_wt = others.totalwgt_lb

    print("Mean weight of first babies:", first_wt.mean())
    print("Mean weight of other babies:", other_wt.mean())

    diff = first_wt.mean() - other_wt.mean()
    print("Weight difference:", diff)

    cohen_d_wt = thinkstats2.CohenEffectSize(first_wt, other_wt)
    print("Cohen's d:", cohen_d_wt)

    cohen_d_prglngth = thinkstats2.CohenEffectSize(firsts.prglngth, others.prglngth)
    print("Cohen's d:", cohen_d_prglngth)
    times = cohen_d_wt/cohen_d_prglngth
    print("Cohen's d of weight is ", "%.2f" % abs(times), " times the Cohen's d of pregnancy length")


def main(script):
    """Tests the functions in this module.

    script: string script name
    """
    live, firsts, others = first.MakeFrames()
    hist = thinkstats2.Hist(live.prglngth)
    investigate_weight(live, firsts, others)

    # test Mode
    mode = Mode(hist)
    print('Mode of preg length', mode)
    assert mode == 39, mode

    # test AllModes
    modes = AllModes(hist)
    assert modes[0][1] == 4693, modes[0][1]

    for value, freq in modes[:5]:
        print(value, freq)

    print('%s: All tests passed.' % script)


if __name__ == '__main__':
    main(*sys.argv)
