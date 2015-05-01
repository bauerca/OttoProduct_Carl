import pandas
import numpy
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.preprocessing import scale, StandardScaler
from sklearn.lda import LDA
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot as plt
from scipy import optimize
import itertools
import sys


def prepare(data, **kw):
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  data[feature_cols] = raw_scaled_features(data, **kw)
  return data

def raw_scaled_features(data, log=False):
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  if log:
    return scale(numpy.log(1.0 + data[feature_cols]))
  else:
    return scale(1.0 * data[feature_cols])

def features(data):
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  return data[feature_cols]

def load_train(log=False):
  """
  Return X, y as numpy arrays. Ugh, pandas.
  """

  train = pandas.read_csv('train.csv')
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  X = train[feature_cols]

  if log:
    X = scale(numpy.log(1.0 + X))
  else:
    X = scale(1.0 * X)

  return X, train['target'].values


def by_class(X, y):
  for i in range(1, 10):
    pass


def plot_pca(kernel=None):
  fig_num = 1

  train = pandas.read_csv('train.csv').groupby('target').head(100)
  y = train['target'].values
  features = raw_scaled_features(train, log=True)

  pca = PCA() #n_components=2)
  if kernel is not None:
    pca = KernelPCA(n_components=10, kernel=kernel, eigen_solver='arpack')

  components = pca.fit_transform(features)

  # Show variance of each component direction.
  #plt.figure(fig_num)
  #plt.plot(pca.explained_variance_ratio_, marker='o')
  #plt.show()

  colors = ['b', 'g', 'r', 'k', 'y', 'm', 'c']
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
      for i, cls in enumerate(class_pair):
        indices = y == ('Class_%d' % cls)
        #print indices
        cx = components[indices, ci]
        cy = components[indices, cj]
        #print '%d points in Class %d' % (len(x), i)
        plt.scatter(cx, cy, c=colors[i], label='Class %d' % cls)
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
  y_true = numpy.asarray(y_true)
  y_prob = numpy.asarray(y_prob)
  error = 0.0

  for i, cls in enumerate(classes):
    error -= numpy.sum(numpy.log(y_prob[y_true == cls, i]))

  return error / float(len(y_true))


def random_subsets(data, n=30):
  subsets = []

  for cls, group in data.groupby('target'):
    indices = numpy.random.randint(0, len(group), n)
    subsets.append(group.iloc[indices])

  subsets = pandas.concat(subsets)
  return subsets


def cluster():
  """
  Find a function f(X) -> X' where X is (n_samples, n_features) and
  X' is (n_samples, m_features) that maximizes the distances between
  the cluster means given by the sets

    S_i = {x_j : x_j in X' and y_j = C_i}

  where C_i is a product class.


  """
  n_features = 93
  feature_cols = ['feat_%d' % i for i in xrange(1, n_features + 1)]

  train = pandas.read_csv('train.csv')
  subsets = random_subsets(train, n=20)

  byclass = subsets.groupby('target')
  classes = byclass.first().index.values
  counts = 1.0 * byclass.size()

  print classes, counts

  def rbf(X, Y, beta):
    """
    Radial basis function kernel.

      exp(-(x - y)^T * diag(beta) * (x - y))

    Arguments:
      X - (N, M)
      Y - (N', M)
      beta - (M,)
        Weights for inner product in exponential.

    Returns
      (N, N')
    """
    res = numpy.zeros((X.shape[0], Y.shape[0]))

    for i, y in enumerate(Y):
      diff = X - y
      res[:,i] = numpy.exp(-numpy.sum(beta * diff**2, axis=1))

    return res

  def f(beta):
    # \sum_{x,x' in C_i} k(x, x') / N_i^2
    cov_sums = pandas.Series(numpy.zeros(len(classes)), index=classes)

    for c, g in byclass:
      gf = features(g).values
      n = float(len(g))
      cov_sums[c] = numpy.sum(rbf(gf, gf, beta)) / n**2

    spread = 0.0

    for ci, gi in byclass:
      gfi = features(gi).values
      ni = float(len(gi))
      for cj, gj in byclass:
        if cj < ci:
          gfj = features(gj).values
          nj = float(len(gj))
          spread += cov_sums[ci] + cov_sums[cj]
          spread -= 2.0 * numpy.sum(rbf(gfi, gfj, beta)) / (ni * nj)

    sum_sigmas = numpy.sum(1.0 - cov_sums)
    res = spread / sum_sigmas
    print 'beta:', beta
    print 'spread:', spread
    print 'sum of sigmas:', sum_sigmas
    print 'ratio:', res
    return -res

  res = optimize.fmin(f, numpy.ones(n_features) / 50., maxiter=1000)
  print res

def svc(n_components=10):
  """
  Train a support vector classifier after dimensionality reduction
  with PCA.

  Each fold takes ~10 min. First fold gave log loss: 0.684875244651
  """

  train = pandas.read_csv('train.csv')
  prepare(train, log=True)

  kpca = KernelPCA(kernel='rbf', n_components=n_components, eigen_solver='arpack')
  kpca.fit(features(train.groupby('target').head(100)))

  y = train['target'].values
  X = numpy.asarray(1.0 * features(train))
  #X = raw_scaled_features(train, log=True)

  folds = StratifiedKFold(train['target'], 10)

  for train_indices, test_indices in folds:
    #print train_indices, test_indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    #pca = PCA(n_components=n_components)
    #X_train = pca.fit_transform(X_train)
    #X_test = pca.transform(X_test)
    #print X_train.shape

    X_train = kpca.transform(X_train)
    X_test = kpca.transform(X_test)

    svc = SVC(kernel='linear', probability=True, verbose=False)
    svc.fit(X_train, y_train)
    y_prob = svc.predict_proba(X_test)

    print log_loss(y_test, y_prob, svc.classes_)



def plot_pca_by_class(prep=True):
  train = pandas.read_csv('train.csv')

  if prep:
    prepare(train)

  i = 1
  pc = numpy.zeros((9, 93))

  for cls, group in train.groupby('target'):
    pca = PCA()
    pca.fit(1.0 * features(group).values)
    pc[i-1] = pca.components_[0]
    plt.figure(1)
    plt.subplot(3, 3, i)
    plt.plot(pca.explained_variance_ratio_, marker='o')
    plt.title(cls)

    #plt.figure(2)
    #inds = numpy.argsort(numpy.abs(pca.components_[0]))
    #X = features(group).values
    #j = 1
    #for ind in inds:
    #  plt.subplot(10, 10, j)
    #  plt.plot(X[:,ind], 'o')
    #  plt.ylim((0, 100))
    #  plt.title('Feature %d' % (ind + 1))
    #  j += 1
    #plt.show()

    i += 1

  plt.show()
  plt.clf()

  plt.pcolormesh(numpy.dot(pc, pc.T), vmin=0.0, vmax=1.0)
  plt.title('Orthogonality between class principal components')
  plt.colorbar()
  plt.show()


def cv_pca_by_class(prep=False, nfolds=10):
  train = pandas.read_csv('train.csv')

  if prep:
    prepare(train, log=False)

  classes = ['Class_%d' % i for i in range(1, 10)] #[1, 2, 3, 4]

  y = train['target'].values
  X = numpy.asarray(1.0 * features(train))
  #X = raw_scaled_features(train, log=True)

  folds = StratifiedKFold(train['target'], nfolds)

  for train_indices, test_indices in folds:
    #print train_indices, test_indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    probs = numpy.zeros((len(X_test), len(classes)))

    for i, cls in enumerate(classes):
      pca = PCA(n_components=None)
      X_train_cls = X_train[y_train == cls]
      #print cls, 'training on', len(X_train_cls)
      pca.fit(X_train_cls)
      proj = numpy.dot(X_test, pca.components_.T)
      #proj = numpy.abs(pca.transform(X_test))
      #print cls, 'scaled projections:', numpy.abs(proj) * pca.explained_variance_ratio_
      #print cls, pca.explained_variance_ratio_
      # For PCA
      probs[:,i] = numpy.sum(numpy.abs(proj) * pca.explained_variance_ratio_, axis=1)
      # For ICA
      #probs[:,i] = numpy.sum(numpy.abs(proj), axis=1)

    probs = probs**4
    probs /= numpy.sum(probs, axis=1).reshape((len(probs), 1))
    print log_loss(y_test, probs, classes)


def corr():
  train = pandas.read_csv('train.csv')
  i = 1

  for cls, group in train.groupby('target'):
    group_corr = features(group).corr()
    plt.subplot(3, 3, i)
    plt.pcolormesh(group_corr.values, vmin=-1.0, vmax=1.0)
    plt.title(cls)
    i += 1

  plt.show()


def tmi():
  train = pandas.read_csv('train.csv')
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  classes = ['Class_%d' % i for i in range(1, 10)] #[1, 2, 3, 4]

  plt.figure(1, figsize=(20., 10.))

  for fi, fj in itertools.combinations(feature_cols, 2):
    i = 1

    for c in classes:
      plt.subplot(3, 3, i)
      indices = (train['target'] == c).values
      train[indices].plot(x=fi, y=fj, kind='scatter', c='b', ax=plt.gca())
      plt.xlim((-10, 100))
      plt.ylim((-10, 100))
      plt.title(c)
      i += 1

    plt.show()
    plt.clf()

    #for ci, cj in itertools.combinations(classes, 2):
    #  indices = (train['target'] == ci).values
    #  train[indices].plot(x=fi, y=fj, kind='scatter', c='b', label=ci, ax=plt.gca())
    #  indices = (train['target'] == cj).values
    #  train[indices].plot(x=fi, y=fj, kind='scatter', c='g', label=cj, ax=plt.gca())
    #  #plt.scatter(train[indices, fi], train[indices, fj], label=ci)
    #  #indices = (train['target'] == cj).values
    #  #plt.scatter(train[indices, fi], train[indices, fj], label=cj)
    #  i += 1

    #  plt.show()
    #  plt.clf()

#plot_pca(kernel='rbf')
#svc(n_components=20)
#corr()
#pca_by_class()
#cv_pca_by_class(prep=False, nfolds=2)
#plot_pca_by_class(prep=False)
#tmi()
cluster()
