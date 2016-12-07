import sys, pickle
import pylab as pl, numpy as np

from sklearn.datasets import make_circles

"""
Quickly generate a linearly nonseparable dataset to illustrate kernel
methods for SVMs. 

Usage:

    $ python generate_dataset.py [-save [PATH]]

If '-save' is set, then the script will store the generated dataset
as a pickle'd file to PATH. PATH defaults to 'dataset.p'.
    
For example:

    $ python generate_dataset.py -save mydataset.p

will generate a dataset, and store it to 'mydataset.p'.
    
"""

def main():
    args = sys.argv[1:]
    
    dataset_path = None
    if args and '-save' in args:
        try: dataset_path = args[args.index('-save') + 1]
        except: dataset_path = 'dataset.p'
        
    # Generate the dataset
    print "...Generating Dataset..."
    X1, Y1 = make_circles(n_samples=800, noise=0.07, factor=0.4)
    frac0 = len(np.where(Y1 == 0)[0]) / float(len(Y1))
    frac1 = len(np.where(Y1 == 1)[0]) / float(len(Y1))
    
    print "Percentage of '0' labels:", frac0
    print "Percentage of '1' labels:", frac1

    # (Optionally) save the dataset to DATASET_PATH
    if dataset_path:
        print "...Saving dataset to {0}...".format(dataset_path)
        pickle.dump((X1, Y1, frac0, frac1), open(dataset_path, 'wb'))

    # Plot the dataset
    print "...Showing dataset in new window..."
    pl.figure(figsize=(10, 8))
    pl.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    pl.subplot(111)
    pl.title("Our Dataset: N=200, '0': {0} '1': {1} ".format(frac0, frac1), fontsize="large")

    pl.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

    pl.show()
    
    print "...Done."
    
if __name__ == '__main__':
    main()
