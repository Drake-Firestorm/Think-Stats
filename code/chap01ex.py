"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import numpy as np
import sys

import nsfg
import thinkstats2

def ReadFemResp(dct_file="2002FemResp.dct", dat_file="2002FemResp.dat.gz", nrows=None):
    """
    Reads the NSFG respondent data.

    :param dct_file: string file name
    :param dat_file: string file name
    :param nrows: None
    :return: DataFrame
    """
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression="gzip", nrows=nrows)
    return df


def validate_pregnum(resp, preg):
    """
    Validates whether pregnant count is the same between resp and preg files.

    :param resp: dataframe
    :param preg: dataframe
    :return: Bool
    """
    preg_map = nsfg.MakePregMap(preg)

    for index, count in resp.pregnum.items():
        caseid = resp.caseid[index]
        if len(preg_map[caseid]) != count:
            print(cid, preg_map.get(cid), resp.loc[resp.caseid == cid].pregnum)
            return False
    return True


def main(script):
    """Tests the functions in this module.

    script: string script name
    """
    print('%s: All tests passed.' % script)


if __name__ == '__main__':
    # main(*sys.argv)

    # Codebook multiplies Index with Count to give the result
    print("Total count by number of Pregnancies")
    resp = ReadFemResp()
    for index, count in resp.pregnum.value_counts().sort_index().items():
        print() if index == 0 else print(index, index*count)

    preg = nsfg.ReadFemPreg()
    print("Count same between resp and preg: ", validate_pregnum(resp, preg))
