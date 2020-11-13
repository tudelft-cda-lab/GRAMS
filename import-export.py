#!/usr/bin/python3

import pickle
import sys

sourcedir = sys.argv[1]
targetdir = sys.argv[2]

import glob
from torch import FloatTensor

for ff in glob.glob(sourcedir + "/*"):

    # pickle.load(open(ff, "rb"))

    with open(sourcedir + "/" + ff.split("/")[-1] ) as f:
        lines = f.readlines()
        array = lines[0].split(",")[0:-1]
        to_numpy = [float(f) for f in array]

    # a = open(targetdir + "/" + ff.split("/")[-1], "wb")
    print(len(to_numpy))
    pickle.dump(FloatTensor(to_numpy), file=open(targetdir + "/" + ff.split("/")[-1], "wb+"))

#import readline; readline.write_history_file('import-export.py')
