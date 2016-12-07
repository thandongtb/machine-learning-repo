import sys, pickle, random, time, os
import pylab as pl, numpy as np
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

DATASET = 'mydataset.p'

class Classifier(object):
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, y):
        raise NotImplementedError
class RandomClassifier(Classifier):
    """ Randomly output class label '0'/'1' with equal probability. """
    def fit(self, X, y):
        pass
    def predict(self, y):
        return [random.randint(0, 1) for _ in y]

def plot_svm(svc, X, y, X_test, y_test, title="", save=None):
    """ Plots the dataset and the learned decision boundary from SVC
    to a figure. If SAVE is given, then the figure is saved to the
    path given by SAVE.
    """
    X0 = X[np.where(y == 0)]
    X1 = X[np.where(y == 1)]

    pl.figure()
    pl.clf()
    pl.scatter(X0[:, 0], X0[:, 1], marker='o', c="#00ffff", zorder=10, cmap=pl.cm.Paired)
    pl.scatter(X1[:, 0], X1[:, 1], marker='o', c='r', zorder=10, cmap=pl.cm.Paired)

    # Circle out the test data
    pl.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    
    pl.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    pl.pcolormesh(XX, YY, Z > 0, cmap=pl.cm.Paired)
    pl.contour(XX, YY, Z, colors=['k', 'k', 'k'],
              linestyles=['--', '-', '--'],
              levels=[-.5, 0, .5])
    pl.title(title)
    
    if save:
        pl.savefig(save)
    else:
        pl.show()
    
def plot_data(X, y, save=None):
    """ Given data X and labels Y, plots the data. If SAVE is given,
    then the figure is saved to the path given by SAVE.
    """
    # Gather stats
    N = len(X)
    frac0 = len(np.where(y == 0)[0]) / float(len(y))
    frac1 = len(np.where(y == 1)[0]) / float(len(y))
    pl.figure()
    pl.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    pl.subplot(111)
    pl.title("Dataset: N={0}, '0': {1} '1': {2} ".format(N, frac0, frac1), fontsize="large")

    X0 = X[np.where(y == 0)]
    X1 = X[np.where(y == 1)]
    
    pl.scatter(X0[:, 0], X0[:, 1], marker='o', c="#00ffff")
    pl.scatter(X1[:, 0], X1[:, 1], marker='o', c='r')

    if save:
        pl.savefig(save)
    else:
        pl.show()
    
def separable_demo():
    """ Generate a linearly-separable dataset D, train a linear SVM on
    D, then output the resulting decision boundary on a figure.
    """
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=200, n_features=2, 
                      centers=((0,0), (4, 4)),
                      cluster_std=1.0)
    plot_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    svc = svm.SVC(class_weight='auto')
    param_grid = {'kernel': ['linear'],
                  'C': [1e0, 1e1, 1e2, 1e3, 1e4]}
    strat_2fold = StratifiedKFold(y_train, k=2)
    print "    Parameters to be chosen through cross validation:"
    for name, vals in param_grid.iteritems():
        if name != 'kernel':
            print "        {0}: {1}".format(name, vals)
    clf = GridSearchCV(svc, param_grid, n_jobs=1, cv=strat_2fold)
    clf.fit(X_train, y_train)
    print "== Best Parameters:", clf.best_params_
    y_pred = clf.predict(X_test)
    acc = len(np.where(y_pred == y_test)[0]) / float(len(y_pred))
    print "== Accuracy:", acc
    print classification_report(y_test, y_pred)
    plot_svm(clf.best_estimator_, X, y, X_test, y_test, 
             title="SVM Decision Boundary, Linear Kernel ({0} accuracy, C={1})".format(acc, clf.best_params_['C']))
    
def train_eval_svm(X_train, X_test, y_train, y_test,
                   X, y,
                   param_grid=None,
                   save=""):
    param_grid = param_grid if param_grid else {}
    svc = svm.SVC(class_weight='auto')
    strat_2fold = StratifiedKFold(y_train, k=2)
    print "    Parameters to be chosen through cross validation:"
    for name, vals in param_grid.iteritems():
        if name != 'kernel':
            print "        {0}: {1}".format(name, vals)
    clf = GridSearchCV(svc, param_grid, n_jobs=1, cv=strat_2fold)
    clf.fit(X_train, y_train)
    print "== Best Params:", clf.best_params_
    print "== Best Score:", clf.best_score_
    y_pred = clf.predict(X_test)
    acc = len(np.where(y_pred == y_test)[0]) / float(len(y_pred))
    print "== Accuracy:", acc
    print classification_report(y_test, y_pred)
    title = ""
    for name, val in clf.best_params_.iteritems():
        if name == 'kernel':
            title = "Kernel={0} \n".format(val) + title
        else:
            title += "{0}={1} ".format(name, val)
    title = "SVM Decision Boundary accuracy={0} ({1})".format(acc, title.strip())
    plot_svm(clf.best_estimator_, X, y, X_test, y_test,
             save=save,
             title=title)
    
def main():
    global DATASET
    args = sys.argv[1:]
    if args and args[0] == '-separable':
        print "======== Running SEPARABLE demo ========"
        separable_demo()
        print "...Done."
        return
    elif args:
        if not os.path.exists(args[0]):
            print "Warning: dataset path {0} doesn't exist. \
Defaulting to path {1}".format(args[0], DATASET)
        else:
            DATASET = args[0]
    print "======== Running NONSEPARABLE demo ========"
    print "...reading dataset from {0}...".format(DATASET)
    X, y, frac0, frac1 = pickle.load(open(DATASET, 'rb'))
    plot_data(X, y)#, save="../imgs/dataset_nonsep.png")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random.randint(0, 99999))

    print "==== Evaluating Random Classifier"
    t = time.time()
    rc = RandomClassifier()
    rc.fit(X_train, y_train)
    y_pred_rc = rc.predict(X_test)
    acc_rc = len(np.where(y_pred_rc == y_test)[0]) / float(len(y_pred_rc))
    print "== Accuracy:", acc_rc
    print classification_report(y_test, y_pred_rc)
    print "==== Finished Random Classifier ({0:.3f} s)\n".format(time.time() - t)
    
    print "==== Evaluating SVM (kernel='linear'), 2-fold cross validation"
    t = time.time()
    param_grid = {'kernel': ['linear'],
                  'C': [1e0, 1e1, 1e2, 1e3, 1e4]}
    train_eval_svm(X_train, X_test, y_train, y_test, 
                   X, y,
                   param_grid=param_grid)
                   #save="../imgs/nonsep_svm_linear.png")
    print "==== Finished Linear SVM ({0:.3f} s)\n".format(time.time() - t)

    print "==== Evaluating SVM (kernel='poly'), 2-fold cross validation"
    # Note: On my Windows (32-bit python) machine, I frequently encounter
    # apparent lock-ups in the training for the poly kernel. I don't
    # know if I'm calling the library incorrectly, or if there is a bug
    # in the sklearn library...?
    t = time.time()
    param_grid = {'kernel': ['poly'],
                  'C': [1e0, 1e1, 1e2, 1e3],
                  'degree': [2, 4],
                  'coef0': [1e0, 1e1, 1e2],
                  'gamma': [1e-3, 1e-2, 1e-1]}
    train_eval_svm(X_train, X_test, y_train, y_test, 
                   X, y,
                   param_grid=param_grid)
                   #save="../imgs/nonsep_svm_poly.png")
    print "==== Finished Polynomial SVM ({0:.3f} s)\n".format(time.time() - t)
    
    print "==== Evaluating SVM (kernel='rbf'), 2-fold cross validation"
    t = time.time()
    param_grid = {'kernel': ['rbf'],
                  'C': [1e0, 1e1, 1e2, 1e3, 1e4],
                  'gamma': [1e-4, 1e-3, 1e-2, 1e-1]}
    train_eval_svm(X_train, X_test, y_train, y_test, 
                   X, y,
                   param_grid=param_grid)
                   #save="../imgs/nonsep_svm_rbf.png")
    print "==== Finished RBF Kernel ({0:.3f} s)\n".format(time.time() - t)
    
    print "==== Evaluating SVM (kernel='sigmoid'), 2-fold cross validation"
    t = time.time()
    param_grid = {'kernel': ['sigmoid'],
                  'gamma': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
                  'C': [1e-1, 1e0, 1e1, 1e2, 1e3, 1e5],
                  'coef0': [-1e4, -1e3, -1e2, -1e1, 1e0, 1e1, 1e2]}
    train_eval_svm(X_train, X_test, y_train, y_test,
                   X, y,
                   param_grid=param_grid)
                   #save="../imgs/nonsep_svm_sigmoid.png")
    print "==== Finished Sigmoid SVM ({0:.3f} s)\n".format(time.time() - t)
                  
if __name__ == '__main__':
    t = time.time()
    main()
    print "...Finished. Total Time: {0:.3f} s".format(time.time() - t)
