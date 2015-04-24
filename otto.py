import pandas
import numpy
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.preprocessing import scale, StandardScaler
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
import itertools
import sys


def raw_scaled_features(data):
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  return scale(1.0 * numpy.log(1.0 + data[feature_cols]))


def plot_pca():
  fig_num = 1

  train = pandas.read_csv('train.csv')
  y = train['target'].values
  features = raw_scaled_features(train)

  pca = PCA() #n_components=2)
  components = pca.fit_transform(features)

  # Show variance of each component direction.
  plt.figure(fig_num)
  plt.plot(pca.explained_variance_ratio_, marker='o')
  plt.show()

  colors = ['k', 'b', 'r', 'g', 'y', 'm', 'c']
  # Component indices to compare. They will be chosen in pairs.
  # Lower index means more variance in that vector direction.
  comp_indices = [0, 1, 2]
  # Classes to compare on the same plot. They will be compared in twos, so
  # for all 9 classes, (9 choose 2) = 36 plots for each pair of components.
  classes = range(1, 10) #[1, 2, 3, 4]

  # Some hopeful cases (good separation in component space) to look out for:
  #   (pca comps: 0, 1) (classes: 6, 7)
  #   (pca comps: 0, 1) (classes: 1, 4) ?
  #   (pca comps: 0, 2) (classes: 2, 7) ?
  #   (pca comps: 0, 2) (classes: 4, 8) ?
  #   (pca comps: 0, 2) (classes: 6, 9) ?
  #   (pca comps: 0, 2) (classes: 8, 9) !
  #   (pca comps: 1, 2) (classes: 4, 5) !

  for ci, cj in itertools.combinations(comp_indices, 2):
    for class_pair in itertools.combinations(classes, 2):
      fig_num += 1
      plt.figure(fig_num)
      for cls in class_pair:
        indices = y == ('Class_%d' % cls)
        #print indices
        cx = components[indices, ci]
        cy = components[indices, cj]
        #print '%d points in Class %d' % (len(x), i)
        plt.scatter(cx, cy, c=colors[cls % 7], label='Class %d' % cls)
      plt.title('PCA components %d and %d' % (ci, cj))
      plt.legend()
      plt.show()


def log_loss(y_true, y_prob, classes):
  """
  Arguments:
    y_true - Array-ish of strings (N,)
      The array of classes.

    y_prob - (N, M) DataFrame
      The columns should be labeled by class
      names occurring in y_true. M is the number of classes

    classes - Array-ish of strings (M,)
      The classes in y_prob sorted according to their column-order
      in y_prob.
  """
  error = 0.0

  for i, cls in enumerate(classes):
    error -= numpy.log(y_prob[y_true == cls, i])

  return error / float(len(y_true))


def cv():
  train = pandas.read_csv('train.csv')
  y = train['target'].values
  X = raw_scaled_features(train)

  folds = StratifiedKFold(train['target'], 10)
  
  for train_indices, test_indices in folds:
    #print train_indices, test_indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
  
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    #print X_train.shape

    svc = SVC(probability=True, verbose=False)
    svc.fit(X_train, y_train)
    y_prob = svc.predict_proba(X_test)

    print log_loss(y_test, y_prob, svc.classes_)


#plot_pca()
#cv()
