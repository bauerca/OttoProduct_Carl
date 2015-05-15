import pandas
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.preprocessing import scale, StandardScaler
from sklearn.lda import LDA
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize, linalg
from nolearn.dbn import DBN
import itertools
import sys


def restrict_to_classes(X, y, classes):
  indices = False
  for c in classes:
    new_indices = (y == c)
    print '%s count: %d' % (c, new_indices.sum())
    indices = indices | new_indices

  return X[indices], y[indices]


def prepare(data, **kw):
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  data[feature_cols] = raw_scaled_features(data, **kw)
  return data

def raw_scaled_features(data, log=False):
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  if log:
    return scale(np.log(1.0 + data[feature_cols]))
  else:
    return scale(1.0 * data[feature_cols])

def features(data):
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  return data[feature_cols]

def load_train(log=False):
  """
  Return X, y as np arrays. Ugh, pandas.
  """

  train = pandas.read_csv('train.csv')
  feature_cols = ['feat_%d' % i for i in xrange(1, 94)]
  X = train[feature_cols]

  if log:
    X = scale(np.log(1.0 + X))
  else:
    X = scale(1.0 * X)

  return X, train['target'].values


def by_class(X, y):
  for i in range(1, 10):
    pass


class CustomRBF:
  """
  Radial basis function kernel.

    exp(-(x - y)^T * diag(beta) * (x - y))
  """

  def __init__(self, weights):
    """
    Arguments:
      weights - (M,)
        Weights for inner product in exponential.
    """
    self.w = np.abs(weights)

  def __call__(self, X, Y):
    """
    Arguments:
      X - (N, M)
      Y - (N', M)

    Returns
      (N, N')
    """
    res = np.zeros((X.shape[0], Y.shape[0]))

    for i, y in enumerate(Y):
      diff = X - y
      res[:,i] = np.exp(-np.sum(self.w * diff**2, axis=1))

    return res


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
  y_true = np.asarray(y_true)
  y_prob = np.asarray(y_prob)
  error = 0.0

  y_prob = np.min([y_prob, (1.0 - 1.0e-15)*np.ones(y_prob.shape)], axis=0)
  y_prob = np.max([y_prob, 1.0e-15*np.ones(y_prob.shape)], axis=0)

  for i, cls in enumerate(classes):
    error -= np.sum(np.log(y_prob[y_true == cls, i]))

  return error / float(len(y_true))


def random_subsets(X, y, n=30):
  classes = np.unique(y)
  #new_X = np.zeros((len(classes) * n, X.shape[1]))
  new_X = []
  new_y = []

  for i, c in enumerate(classes):
    group = np.random.permutation(X[y == c])[:n]
    m = len(group)
    new_X.append(group)
    #new_X[i*m:(i+1)*n, :] = 
    new_y += m * [c]

  return np.concatenate(new_X), np.asarray(new_y)


def df_random_subsets(data, n=30):
  subsets = []

  for cls, group in data.groupby('target'):
    indices = np.random.randint(0, len(group), n)
    subsets.append(group.iloc[indices])

  subsets = pandas.concat(subsets)
  return subsets


def plot_clusters(X, y, sigmas=True):
  colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
  markers = ['o', 's', '^']

  for i, c in enumerate(np.unique(y)):
    #plt.subplot(3, 3, i)
    Xc = X[y == c]
    plt.scatter(Xc[:,0], Xc[:,1], c=colors[i % 7], marker=markers[i % 3], label=c)

    if sigmas:
      center = np.mean(Xc, axis=0)
      r = np.mean(np.sqrt(np.sum((Xc - center)**2, axis=1)))
      plt.gca().add_patch(Circle(center, r, fill=False, ec=colors[i % 7]))

  plt.legend()

def plot3d_clusters(X, y, sigmas=True, ax3d=None):
  colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
  markers = ['o', 's', '^']
  fig = plt.figure()
  ax3d = fig.add_subplot(111, projection='3d')

  for i, c in enumerate(np.unique(y)):
    #plt.subplot(3, 3, i)
    Xc = X[y == c]
    ax3d.scatter(Xc[:,0], Xc[:,1], Xc[:,2], c=colors[i % 7], marker=markers[i % 3], label=c)

  plt.legend()



def eigspread(X, y, Q=10):
  N, M = X.shape
  classes = np.unique(y)

  #pca = PCA(n_components=Q)
  #svc_cv(pca.fit_transform(Xall), yall, groupsize=200, nfolds=10, kernel='rbf')

  Nc = []
  Xc = []
  X_sigma = np.zeros((M, M))
  X_delta = np.zeros((M, M))

  for i, ci in enumerate(classes):
    Xci = X[y == ci]
    Nci = float(len(Xci))
    cross = np.zeros((M, M))

    #for xi in Xci:
    #  for xj in Xci:
    #    cross += np.outer(xi, xj)
    for k in xrange(M):
      for l in xrange(M):
        cross[k,l] = np.sum(np.outer(Xci[:,k], Xci[:,l]))

    X_sigma += cross / Nci**2

    # Cache for cross-cluster stuff
    Xc.append(Xci)
    Nc.append(Nci)

    for j in xrange(i):
      print 'X_delta (%d, %d)' % (i, j)
      cross = np.zeros((M, M))

      Xcj = Xc[j]
      Ncj = Nc[j]

      #for xi in Xci:
      #  for xj in Xcj:
      #    cross += np.outer(xi, xj)
      for k in xrange(M):
        for l in xrange(M):
          cross[k,l] = np.sum(np.outer(Xci[:,k], Xcj[:,l]))

      # Separate all
      X_delta += (1.0 / (Nci * Ncj)) * cross

  X_delta += X_delta.T

  w, V = linalg.eigh(X_sigma, X_delta, eigvals=(M-Q, M-1))

  print ('Eigspread eigenvalues:'), w
  return V

def run_eigspread():
  train = pandas.read_csv('train.csv')
  #Xall = np.log(1.0 + features(train).values)
  #Xall = np.concatenate([Xall, np.ones((len(Xall), 1))], axis=1)
  #Xall = scale(1.0 * features(train).values)
  Xall = 1.0 * features(train).values
  yall = train['target'].values
  X, y = random_subsets(Xall, yall, n=200)

  V = eigspread(X, y, Q=10)

  Zall = np.dot(Xall, V)
  Z = np.dot(X, V)

  plot_clusters(Zall[:,-2:], yall)
  #plot3d_clusters(np.dot(X, V[:,-3:]), y)
  plt.show()


def chis(X, y, save=False, weights=None):
  if isinstance(X, str):
    X_delta = np.load('X-delta.npy')
    X_sigma = np.load('X-sigma.npy')
    X = np.load(X)
    y = np.load(y)
    return X, y, X_delta, X_sigma

  N, M = X.shape
  classes = np.sort(np.unique(y))

  if weights is None:
    weights = {c: 1.0 for c in classes}

  #pca = PCA(n_components=Q)
  #svc_cv(pca.fit_transform(Xall), yall, groupsize=200, nfolds=10, kernel='rbf')

  Nc = []
  Xc = []
  cov = []
  X_sigma = np.zeros((M, M))
  X_delta = np.zeros((M, M))

  # Spread of all points.
  #means = np.zeros((len(classes), M))
  #for i, ci in enumerate(classes):
  #  means[i] = np.mean(X[y == ci], axis=0)
  #
  #X_sigma = np.dot(means.T, means)
  #cross = np.zeros((M, M))

  #for k in xrange(M):
  #  for l in xrange(M):
  #    cross[k,l] = np.sum(np.outer(means[:,k], means[:,l]))

  #X_sigma -= cross / float(len(means))
  #print X_sigma
  #sys.exit()
  # /Spread of all points

  for i, ci in enumerate(classes):
    Xci = X[y == ci]
    Nci = float(len(Xci))
    covi = np.dot(Xci.T, Xci) / Nci
    cross = np.zeros((M, M))

    #for xi in Xci:
    #  for xj in Xci:
    #    cross += np.outer(xi, xj)
    for k in xrange(M):
      for l in xrange(M):
        cross[k,l] = np.sum(np.outer(Xci[:,k], Xci[:,l]))

    X_sigma += weights[ci] * (covi - cross / Nci**2)

    # Cache for cross-cluster stuff
    Xc.append(Xci)
    Nc.append(Nci)
    cov.append(covi)

    for j in xrange(i):
      print 'X_delta (%d, %d)' % (i, j)
      cross = np.zeros((M, M))

      Xcj = Xc[j]
      Ncj = Nc[j]

      #for xi in Xci:
      #  for xj in Xcj:
      #    cross += np.outer(xi, xj)
      for k in xrange(M):
        for l in xrange(M):
          cross[k,l] = np.sum(np.outer(Xci[:,k], Xcj[:,l]))

      # Separate all
      weight = np.sqrt(weights[ci] * weights[classes[j]])
      X_delta += weight * (cov[i] + cov[j] - (2.0 / (Nci * Ncj)) * cross)

  X_delta += X_delta.T
  X_sigma += np.diag(1.0e-15 * np.trace(X_sigma) * np.ones(M))

  if save:
    np.save('X-delta', X_delta)
    np.save('X-sigma', X_sigma)
    np.save('X', X)
    np.save('y', y)

  return X, y, X_delta, X_sigma

# Uncomment to save chi matrices.
#chis(from_cache=False, save=True, n=1000)
#sys.exit()

class Eigencluster:
  def __init__(self, width=10, k=10):
    self.width = width
    self.k = k

  def fit(self, X, y, width=10):
    classes = np.sort(np.unique(y))
    self.classes = classes
    Nc = len(classes)
    N, M = X.shape
    Ns = []
    Xs = []
    covs = []
    sigmas = []

    for i, ci in enumerate(classes):
      print 'X_sigma %s' % ci
      Xi = X[y == ci]
      Ni = float(len(Xi))
      cov = np.dot(Xi.T, Xi) / Ni
      cross = np.zeros((M, M))

      #for xi in Xci:
      #  for xj in Xci:
      #    cross += np.outer(xi, xj)
      for k in xrange(M):
        for l in xrange(M):
          cross[k,l] = np.sum(np.outer(Xi[:,k], Xi[:,l]))

      sigma = cov - cross / Ni**2

      # Cache for cross-cluster stuff
      Xs.append(Xi)
      Ns.append(Ni)
      covs.append(cov)
      sigmas.append(sigma)


    # Map (ci, cj) tuple to 1-form
    self.forms = {}

    for i in xrange(Nc):
      ci = classes[i]
      Xi = Xs[i]
      Ni = Ns[i]

      for j in xrange(i):
        cj = classes[j]
        print 'X_delta (%s, %s)' % (classes[i], classes[j])
        Xj = Xs[j]
        Nj = Ns[j]

        cross = np.zeros((M, M))
        for k in xrange(M):
          for l in xrange(M):
            cross[k,l] = np.sum(np.outer(Xi[:,k], Xj[:,l]))

        X_sigma = sigmas[i] + sigmas[j]
        X_sigma += np.diag(1.0e-15 * np.trace(X_sigma) * np.ones(M))

        X_delta = covs[i] + covs[j] - (2.0 / (Ni * Nj)) * cross
        X_delta += X_delta.T

        w, V = linalg.eigh(X_delta, X_sigma, eigvals=(M-2, M-1))
        v = V[:,-1]

        form = {'v': v}
        form[ci] = np.sort(np.dot(Xi, v))
        form[cj] = np.sort(np.dot(Xj, v))
        self.forms[(ci, cj)] = form

        # Visualize
        if True:
          #Z = np.dot(Xall, V[:,-1]) # 1-d array
          #plt.hist([Zall[yall == ci], Zall[yall == cj]], bins=50, label=[ci, cj])
          plt.scatter(np.dot(Xi, V[:,-1]), np.dot(Xi, V[:,-2]), c='b', label=ci)
          plt.scatter(np.dot(Xj, V[:,-1]), np.dot(Xj, V[:,-2]), c='g', label=cj)
          plt.legend()
          plt.show()
          plt.clf()


  def predict_proba(self, X):
    """
    For every class, there is a set of hyperplanes around its
    cluster of training points. To find the probability that a
    new sample is of class c, loop through all the pairs that include
    c; for each pair, calculate the probability that it is c based
    purely on that hyperplane.
    """

    N = len(X)
    Nc = len(self.classes)
    p = np.ones((N, Nc, Nc))
    classes = list(self.classes)

    for (ci, cj), form in self.forms.iteritems():
      print 'Probability (%s, %s)' % (ci, cj)
      zi, zj = form[ci], form[cj]
      z = np.dot(X, form['v'])

      di = np.sum(np.subtract.outer(z, zi)**2, axis=1)
      dj = np.sum(np.subtract.outer(z, zj)**2, axis=1)
      rij = di / dj
      rji = dj / di
      rs = rij + rji

      wi = rji / rs
      wj = rij / rs
      #Np = 10
      #print zip(wi[:Np], wj[:Np])
      #sys.exit()

      #zij = np.concatenate([zi, zj])
      #sort_inds = np.argsort(zij)
      #cij = np.concatenate([len(zi) * [ci], len(zj) * [cj]])[sort_inds]
      #zij = zij[sort_inds]

      # nearest neighbor
      #inserts = np.searchsorted(zij, z)
      #lower = inserts - self.k / 2
      #lower[lower < 0] = 0
      #upper = lower + self.k
      #upper[upper > len(zij)] = len(zij)

      #wi = np.zeros(len(z))
      #wj = np.zeros(len(z))

      #for m, (l, u) in enumerate(zip(lower, upper)):
      #  wi[m] = (cij[l:u] == ci).sum() / float(u - l)
      #  wj[m] = 1.0 - wi[m]
      #  #print wi[m], wj[m]

      # Find the average density of points along the line
      #dz = np.mean(zij[1:] - zij[:-1])
      #sigma = 2.0 * (self.width * dz)**2
      #hw = 0.5 * self.width * dz

      #wi = np.zeros(len(z))
      #wj = np.zeros(len(z))

      #for n, zn in enumerate(z):
      #  for zk, wk in ((zi, wi), (zj, wj)):
      #    zshift = zn - zk
      #    zrange = zshift[(zshift > -hw) & (zshift < hw)]
      #    wk[n] = np.sum(0.5 * (1.0 + np.cos(np.pi * zrange / hw)))
      #    #print 'sum:', wk[n], 'stats:', len(zrange)

      #ws = wi + wj
      #ws_max = np.amax(ws)
      #wr = ws / ws_max

      #wi = 0.5 * (1.0 - wr) + wi * wr / ws
      #wi[np.isnan(wi)] = 0.5
      #wj = 1.0 - wi

      #wi = np.sum(np.exp(-np.subtract.outer(z, zi)**2 / sigma), axis=1)
      #wj = np.sum(np.exp(-np.subtract.outer(z, zj)**2 / sigma), axis=1)
      #wi = np.nan_to_num(wi)
      #wj = np.nan_to_num(wj)
      #wi[wi < 1.0e-8] = 0.0
      #wj[wj < 1.0e-8] = 0.0
      #ws = wi + wj
      #wi /= ws
      #wj /= ws
      #if np.any(np.isnan(wi)) or np.any(np.isnan(wj)):
      #  print np.amin(ws)
      #  print wi, wj
      #  sys.exit()

      #eps = 1.0e-15
      ici, icj = classes.index(ci), classes.index(cj)
      p[:,ici,icj] = wi
      p[:,icj,ici] = wj
      #p[p < eps] = eps

      if False:
        plt.scatter(z, wi, label='Prob. %s' % ci, c='b')
        plt.scatter(z, wj, label='Prob. %s' % cj, c='g')
        plt.legend()
        plt.show()
        plt.clf()

    #return p / p.sum(axis=1, keepdims=True)
    p = np.amin(p, axis=2)
    ps = p.sum(axis=1)
    for e in xrange(-10, 3):
      print 'p sum < 10^%d: %d' % (e, (ps < 10.0**e).sum())

    return p / p.sum(axis=1, keepdims=True)

def eigencluster2():
  train = pandas.read_csv('train.csv')
  Xall = np.log(1.0 + features(train).values)
  #Xall = np.concatenate([Xall, np.ones((len(Xall), 1))], axis=1)
  #Xall = scale(1.0 * features(train).values)
  #Xall = 1.0 * features(train).values
  yall = train['target'].values

  #print np.sum(Xall, axis=0)
  #sys.exit()

  # All equally likely gives 2.1 or something.
  print log_loss(yall, np.ones((len(yall), 9)) / float(9), np.unique(yall))
  print log_loss(yall, np.zeros((len(yall), 9)), np.unique(yall))

  X, y = Xall, yall
  #X, y = restrict_to_classes(X, y, ['Class_6', 'Class_7'])
  X, y = random_subsets(X, y, n=500)
  print 'Eigencluster2 with %d samples' % X.shape[0]

  ec = Eigencluster(k=10, width=10)
  cv(X, y, [ec])
  sys.exit()


  ec.fit(X, y)
  p = ec.predict_proba(Xall)
  y_pred = [ec.classes[i] for i in np.argmax(p, axis=1)]
  print confusion_matrix(yall, y_pred, ec.classes)

  #print p
  print log_loss(yall, p, ec.classes)


def eigencluster():
  train = pandas.read_csv('train.csv')
  Xall = np.log(1.0 + features(train).values)
  #Xall = np.concatenate([Xall, np.ones((len(Xall), 1))], axis=1)
  #Xall = scale(1.0 * features(train).values)
  #Xall = 1.0 * features(train).values
  yall = train['target'].values
  classes = np.sort(np.unique(yall))

  # class weights for clustering
  #weights = np.zeros(9)
  #weights[7:9] = 1.0e1
  #weights /= np.sum(weights)
  #weights = {'Class_%d' % (i+1): w for (i, w) in enumerate(weights)}
  #print weights
  weights = None

  X, y = random_subsets(Xall, yall, n=500)
  print 'Eigencluster with %d samples' % X.shape[0]

  #pca = PCA(n_components=Q)
  #svc_cv(pca.fit_transform(Xall), yall, groupsize=200, nfolds=10, kernel='rbf')

  Q = 2
  O = 1
  P = 10

  for order in xrange(O):
    #X, y, X_delta, X_sigma = chis('X.npy', 'y.npy')
    X, y, X_delta, X_sigma = chis(X, y, save=False, weights=weights)
    M = X_delta.shape[0]

    w, V = linalg.eigh(X_delta, X_sigma, eigvals=(M-Q, M-1))

    print ('Order %d eigenvalues:' % order), w
    Zall = np.dot(Xall, V)
    Z = np.dot(X, V)

    # Lift to Fourier space
    X = np.zeros((len(Z), 0 + 2*P*Q))
    #X[:,:Q] = Z
    Xall = np.zeros((len(Zall), 0 + 2*P*Q))
    #Xall[:,:Q] = Zall

    #for p in xrange(1, P+1):
      #X[:,(2*p-1)*Q:2*p*Q] = np.sin(p*Z)
      #X[:,2*p*Q:(2*p+1)*Q] = np.cos(p*Z)
      #Xall[:,(2*p-1)*Q:2*p*Q] = np.sin(p*Zall)
      #Xall[:,2*p*Q:(2*p+1)*Q] = np.cos(p*Zall)
    for p in xrange(P):
      X[:,2*p*Q:(2*p+1)*Q] = np.sin(p*Z)
      X[:,(2*p+1)*Q:2*(p+1)*Q] = np.cos(p*Z)
      Xall[:,2*p*Q:(2*p+1)*Q] = np.sin(p*Zall)
      Xall[:,(2*p+1)*Q:2*(p+1)*Q] = np.cos(p*Zall)


    if True:
      # Lift to polynomial space
      X = np.zeros((len(Z), Q + Q * (Q + 1) / 2))
      X[:,:Q] = Z
      Xall = np.zeros((len(Zall), Q + Q * (Q + 1) / 2))
      Xall[:,:Q] = Zall

      for i, z in enumerate(Z):
        z2 = np.outer(z, z)
        k = Q
        for j, row in enumerate(z2):
          X[i,k:k+Q-j] = row[j:]
          k += Q-j

      for i, z in enumerate(Zall):
        z2 = np.outer(z, z)
        k = Q
        for j, row in enumerate(z2):
          Xall[i,k:k+Q-j] = row[j:]
          k += Q-j


  #knn_cv(Zall, yall, groupsize=2000, nfolds=5, k=20)
  #svc_cv(Zall, yall, groupsize=2000, nfolds=5, kernel='rbf', C=10.0)
  #cluster_cv(Zall, yall, groupsize=500, nfolds=5)

  plt.hist([Zall[yall == split[0], -1], Zall[yall == split[1] ,-1]], bins=500)
  plt.show()

  plot_clusters(Zall[:,-2:], yall)
  #plot3d_clusters(np.dot(X, V[:,-3:]), y)
  plt.show()


def cv(X, y, clfs, groupsize=None, nfolds=10, verbose=True):
  """
  Cross validate classifiers.
  """

  if groupsize is not None:
    X, y = random_subsets(X, y, n=groupsize)

  folds = StratifiedKFold(y, nfolds)
  classes = np.sort(np.unique(y))
  Nc = len(classes)
  losses = []

  for train_indices, test_indices in folds:
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    p = np.zeros((len(y_test), len(classes)))
    for clf in clfs:
      clf.fit(X_train, y_train)
      #y_pred = clf.predict(X_test)
      #print confusion_matrix(y_test, y_pred)
      p += clf.predict_proba(X_test)
    p /= np.sum(p, axis=1).reshape((len(p), 1))
    loss = log_loss(y_test, p, classes)
    if verbose:
      print loss
    losses.append(loss)

  return losses


def opt_svc():
  train = pandas.read_csv('train.csv')
  #Xall = scale(1.0 * features(train).values)
  Xall = scale(np.log(1.0 + features(train).values))
  yall = train['target'].values
  N, M = Xall.shape

  X, y = random_subsets(Xall, yall, n=1000)
  cls = 'Class_1'
  y[y != cls] == 'other'

  def f(b, X, y, groupsize=None):
    svc = SVC(C=b[0], gamma=b[1], kernel='rbf', probability=True)
    res = np.mean(cv(X, y, [svc], groupsize=groupsize, nfolds=2))
    print 'C=%f, gamma=%f, log loss=%f' % (b[0], b[1], res)
    return res

  res = optimize.fmin(f, [1.0, 0.001], args=(X, y), maxiter=1000)
  print 'Optimum SVC parameters on full dataset for %s vs rest:' % cls
  print f(res, Xall, yall, groupsize=None)


def jumble():
  train = pandas.read_csv('train.csv')
  Xall = scale(1.0 * features(train).values)
  #Xall = scale(np.log(1.0 + features(train).values))
  yall = train['target'].values
  clfs = [
    #KNeighborsClassifier(n_neighbors=3),
    # C=8.0, gamma=0.004 for all together.
    SVC(C=8.0, gamma=0.004, kernel='rbf', probability=True, class_weight={'Class_2':0.1, 'Class_3':0.1})
    #GaussianNB()
    #SGDClassifier(loss='log', alpha=0.01, penalty='l2', n_iter=20)
  ]
  cv(Xall, yall, clfs, groupsize=1000, nfolds=4)



def grid():
  train = pandas.read_csv('train.csv')
  #Xall = scale(1.0 * features(train).values)
  Xall = scale(np.log(1.0 + features(train).values))
  yall = train['target'].values
  classes = np.sort(np.unique(yall))
  classes = ['Class_2', 'Class_3']

  #X, y = random_subsets(Xall, yall, n=2000)
  X, y = Xall, yall

  for i, ci in enumerate(classes):
    for j in xrange(i):
      cj = classes[j]

      print 'Grid search for (%s, %s)' % (ci, cj)
      indices = (y == ci) | (y == cj)
      Xij = X[indices]
      yij = y[indices]
      min_err = 100.0

      for C_exp in [0.7]:
        for C in [10.0**C_exp]: #, 10.0**(-C_exp)]:
          #for gamma_exp in [-4, -3, -2, -1]: #xrange(0, 3):
          for gamma_exp in [-2.2, -2, -1.8]: #xrange(0, 3):
            for gamma in [10.0**gamma_exp]: #, 10.0**(-gamma_exp)]:
              svc = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
              errors = cv(Xij, yij, [svc], groupsize=None, nfolds=2, verbose=False)
              if True or np.mean(errors) < min_err:
                print 'C = %f, gamma = %f' % (C, gamma)
                print ' ', errors
                min_err = np.mean(errors)

      print '\n\n'


def brute():
  p = {}
  p[('Class_2', 'Class_1')] = [10.0, 0.01]

  p[('Class_3', 'Class_1')] = [10.0, 0.01]
  p[('Class_3', 'Class_2')] = [10.0, 0.01]

  p[('Class_4', 'Class_1')] = [10.0, 0.01]
  p[('Class_4', 'Class_2')] = [10.0, 0.01]
  p[('Class_4', 'Class_3')] = [10.0, 0.01]

  p[('Class_5', 'Class_1')] = [10.0, 0.01]
  p[('Class_5', 'Class_2')] = [10.0, 0.01]
  p[('Class_5', 'Class_3')] = [10.0, 0.01]
  p[('Class_5', 'Class_4')] = [10.0, 0.01]

  p[('Class_6', 'Class_1')] = [10.0, 0.01]
  p[('Class_6', 'Class_2')] = [10.0, 0.01]
  p[('Class_6', 'Class_3')] = [10.0, 0.01]
  p[('Class_6', 'Class_4')] = [10.0, 0.01]
  p[('Class_6', 'Class_5')] = [10.0, 0.01]

  p[('Class_7', 'Class_1')] = [10.0, 0.01]
  p[('Class_7', 'Class_2')] = [10.0, 0.01]
  p[('Class_7', 'Class_3')] = [10.0, 0.01]
  p[('Class_7', 'Class_4')] = [10.0, 0.01]
  p[('Class_7', 'Class_5')] = [10.0, 0.01]
  p[('Class_7', 'Class_6')] = [10.0, 0.01]

  p[('Class_8', 'Class_1')] = [10.0, 0.01]
  p[('Class_8', 'Class_2')] = [10.0, 0.01]
  p[('Class_8', 'Class_3')] = [10.0, 0.01]
  p[('Class_8', 'Class_4')] = [10.0, 0.01]
  p[('Class_8', 'Class_5')] = [10.0, 0.01]
  p[('Class_8', 'Class_6')] = [10.0, 0.01]
  p[('Class_8', 'Class_7')] = [10.0, 0.01]

  p[('Class_9', 'Class_1')] = [10.0, 0.01]
  p[('Class_9', 'Class_2')] = [10.0, 0.01]
  p[('Class_9', 'Class_3')] = [10.0, 0.01]
  p[('Class_9', 'Class_4')] = [10.0, 0.01]
  p[('Class_9', 'Class_5')] = [10.0, 0.01]
  p[('Class_9', 'Class_6')] = [10.0, 0.01]
  p[('Class_9', 'Class_7')] = [10.0, 0.01]
  p[('Class_9', 'Class_8')] = [10.0, 0.01]

  train = pandas.read_csv('train.csv')
  #Xall = scale(1.0 * features(train).values)
  Xall = scale(np.log(1.0 + features(train).values))
  yall = train['target'].values
  classes = np.sort(np.unique(yall))
  Nc = len(classes)

  X, y = random_subsets(Xall, yall, n=1000)
  #X, y = Xall, yall

  probs = numpy.ones((len(X), Nc, Nc))

  for i, ci in enumerate(classes):
    for j in xrange(i):
      cj = classes[j]
      C, gamma = p[(ci, cj)]
      svc = SVC(C=C, gamma=gamma, probability=True, kernel='rbf')
      indices = (y == ci) | (y == cj)
      Xij = X[indices]
      yij = y[indices]

      for train_indices, test_indices in StratifiedKFold(yij, nfolds):
        Xij_train = Xij[train_indices]
        yij_train = yij[train_indices]
        Xij_test = Xij[test_indices]
        yij_test = yij[test_indices]
        svc.fit(Xij_train, yij_train)
        yij_prob = svc.predict_proba(Xij_test)
        probs[:,i,j] = yij_prob[:,0]
        probs[:,j,i] = yij_prob[:,1]


def knn():
  train = pandas.read_csv('train.csv')
  #Xall = scale(1.0 * features(train).values)
  Xall = scale(np.log(1.0 + features(train).values))
  yall = train['target'].values

  X, y = restrict_to_classes(Xall, yall, ['Class_2', 'Class_3'])
  #X, y = random_subsets(X, y, n=2000)

  clf = KNeighborsClassifier(n_neighbors=35)
  cv(X, y, [clf], nfolds=3)



def cluster_cv(X, y, groupsize=200, nfolds=10, power=2):
  """
  Cross validate eigencluster using a nearest neighbor classification
  on the transformed dataset.
  """

  X, y = random_subsets(X, y, n=groupsize)
  folds = StratifiedKFold(y, nfolds)
  classes = np.unique(y)
  Nc = len(classes)

  for train_indices, test_indices in folds:
    print len(train_indices), len(test_indices)
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    p = np.zeros((len(X_test), Nc))

    for i, x in enumerate(X_test):
      # Inverse distances between test sample and training samples
      invd = 1.0 / np.sum((X_train - x)**power, axis=1)
      sums = np.zeros(Nc)

      for j, c in enumerate(classes):
        sums[j] = np.sum(invd[y_train == c])

      p[i,:] = sums / np.sum(sums)
      #if y_test[i] == 'Class_1':
        #print 'CV sample %d' % i
        #print ' ', y_test[i]
        #print ' ', p[i]
        #print p[i]

    print log_loss(y_test, p, classes)


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
  subsets = df_random_subsets(train, n=200)

  byclass = subsets.groupby('target')
  classes = byclass.first().index.values
  counts = 1.0 * byclass.size()

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
    res = np.zeros((X.shape[0], Y.shape[0]))

    for i, y in enumerate(Y):
      diff = X - y
      res[:,i] = np.exp(-np.sum(np.abs(beta) * diff**2, axis=1))

    return res


  class Func:
    def __call__(self, beta):
      # \sum_{x,x' in C_i} k(x, x') / N_i^2
      cov_sums = pandas.Series(np.zeros(len(classes)), index=classes)
      # \sum_{x in C_i} k(x, x) / N_i
      diag_sums = pandas.Series(np.zeros(len(classes)), index=classes)

      for c, g in byclass:
        gf = features(g).values
        n = float(len(g))
        K = self.kernel(gf, gf, beta)
        cov_sums[c] = np.sum(K) / n**2
        diag_sums[c] = np.sum(np.diag(K)) / n

      spread = 0.0

      for ci, gi in byclass:
        gfi = features(gi).values
        ni = float(len(gi))
        for cj, gj in byclass:
          if cj < ci:
            gfj = features(gj).values
            nj = float(len(gj))
            spread += cov_sums[ci] + cov_sums[cj]
            spread -= 2.0 * np.sum(self.kernel(gfi, gfj, beta)) / (ni * nj)

      sum_sigmas = np.sum(diag_sums - cov_sums)
      res = spread / sum_sigmas
      print 'beta:', beta
      print 'spread:', spread
      print 'sum of sigmas:', sum_sigmas
      print 'ratio:', res
      return -res


  class LinearReduction(Func):
    def __init__(self, dim=2, n_features=93):
      self.dim = dim
      self.n_features = n_features

    def kernel(self, X, Y, beta):
      P = beta.reshape((self.dim, self.n_features)).T
      return np.dot(np.dot(X, P), np.dot(Y, P).T)

    def transform(self, X, beta):
      P = beta.reshape((self.dim, self.n_features)).T
      return np.dot(X, P)

    def w0(self):
      return 2.0 * (np.random.random(self.dim * self.n_features) - 0.5)

    def plot(self, beta, sigmas=False):
      colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
      markers = ['o', 's', '^']
      P = beta.reshape((self.dim, self.n_features)).T
      i = 0
      for c, g in byclass:
        X = np.dot(features(g).values, P)
        #plt.subplot(3, 3, i)
        plt.scatter(X[:,0], X[:,1], c=colors[i % 7], marker=markers[i % 3], label=c)

        if sigmas:
          center = np.mean(X, axis=0)
          r = np.mean(np.sqrt(np.sum((X - center)**2, axis=1)))
          plt.gca().add_patch(Circle(center, r, fill=False, ec=colors[i % 7]))

        i += 1

      plt.legend()


  class PCALinearReduction(LinearReduction):
    """
    Reduce input-feature dimensionality with PCA.
    """

    def __init__(self, X, dim=2, n_components=10):
      LinearReduction.__init__(self, dim=dim, n_features=n_components)
      # (n_feat, n_comp)
      self._comps = PCA(n_components=n_components).fit(X).components_.T

    def transform(self, X, beta):
      Xp = np.dot(X, self._comps)
      return LinearReduction.transform(self, Xp, beta)

    def kernel(self, X, Y, beta):
      Xp = np.dot(X, self._comps)
      Yp = np.dot(Y, self._comps)
      return LinearReduction.kernel(self, Xp, Yp, beta)


  X = 1.0 * features(train).values
  y = train['target'].values

  print 'Linear SVC on 10D PCA data'
  Xpca = PCA(n_components=10).fit_transform(X)
  lsvc_cv(Xpca, y, groupsize=200, nfolds=10)

  lrpca = PCALinearReduction(X, dim=9, n_components=10)
  w0 = lrpca.w0()
  res = optimize.fmin(lrpca, w0, maxiter=10000)
  lsvc_cv(lrpca.transform(X, res), y, groupsize=200, nfolds=10)
  sys.exit()

  #plt.subplot(121)
  #plot_clusters(lrpca.transform(X, w0), y)
  #plt.subplot(122)
  #plot_clusters(lrpca.transform(X, res), y)
  #plt.show()


  lr = LinearReduction(dim=2)

  print 'Linear SVC on 2D PCA data'
  Xpca = PCA(n_components=2).fit_transform(X)
  plt.subplot(121)
  plot_clusters(Xpca, y)
  lsvc_cv(Xpca, y, groupsize=200, nfolds=10)

  print 'Linear SVC on 2D optcluster data'
  beta_fromPCA_100k = np.loadtxt('cluster_linear_fromPCA_fmin100k.txt')
  Xlr = lr.transform(X, beta_fromPCA_100k)
  plt.subplot(122)
  plot_clusters(Xlr, y)
  lsvc_cv(Xlr, y, groupsize=200, nfolds=10)

  plt.figure(2)
  lr.plot(beta_fromPCA_100k, sigmas=True)
  plt.show()

  #beta0 = 2.0 * (np.random.random(2*93) - 0.5)

  pca = PCA(n_components=2)
  pca.fit(1.0 * features(subsets).values)
  beta0 = pca.components_.flatten()

  res = optimize.fmin(lr, beta0, maxiter=100000)
  print res

  #plt.figure(1)
  plt.subplot(121)
  lr.plot(beta0)
  plt.title('Initial')

  #plt.figure(2)
  plt.subplot(122)
  lr.plot(res)
  plt.title('optimized')

  plt.show()
  sys.exit()


  # spread: 7.57944106898
  # sum of sigmas: 7.82678661884
  # ratio: 0.968397560595
  beta0 = [
   1.97937904e-02,  2.09697373e-02,  1.97602619e-02,  2.29713609e-02,
   1.89022561e-02,  2.11266579e-02,  2.06388922e-02,  1.97510513e-02,
   1.96269322e-02,  2.01458774e-02,  1.96956710e-02,  2.07461472e-02,
   1.92185214e-02,  1.90597482e-02,  2.09208783e-02,  1.88683077e-02,
   2.05427241e-02,  2.38246369e-02,  1.74301233e-02,  2.01794112e-02,
   2.05120477e-02,  1.99764439e-02,  2.03300554e-02,  1.83119249e-02,
   2.02714147e-02,  2.05032421e-02,  1.94780292e-02,  2.02757550e-02,
   2.00939729e-02,  2.00333158e-02,  2.04026553e-02,  2.08000677e-02,
   2.02023788e-02, -5.90571278e-07,  1.98071893e-02,  2.24698584e-02,
   2.05974852e-02,  1.66059498e-02,  2.24482815e-02,  1.99106743e-02,
   1.83773552e-02,  2.04635946e-02,  1.97173055e-02,  2.05034796e-02,
   2.01263825e-02,  2.05656383e-02,  2.40093408e-02,  1.93932056e-02,
   2.04805362e-02,  2.04506010e-02,  2.06895361e-02,  2.03879553e-02,
   2.06291349e-02,  2.08500963e-02,  1.82011569e-02,  1.95928625e-02,
   2.16621080e-02,  2.00374440e-02,  1.98195672e-02,  1.97160556e-02,
   2.04650218e-02,  2.04733047e-02,  2.00538932e-02,  1.78187375e-02,
   2.08613772e-02,  1.98952890e-02,  1.85132112e-02,  2.06320461e-02,
   1.91273549e-02,  2.05474741e-02,  1.74197293e-02,  1.98496756e-02,
   2.44594952e-02,  1.96383596e-02,  2.01792334e-02,  2.01987450e-02,
   1.97733793e-02,  2.07221204e-02,  2.42708639e-02,  2.04267702e-02,
   2.03453017e-02,  2.02582442e-02,  2.10294289e-02,  1.89568396e-02,
   1.98266333e-02,  2.05653820e-02,  2.07231908e-02,  2.06521860e-02,
   2.08141757e-02,  2.02049611e-02,  2.00691448e-02,  1.94086036e-02,
   1.99469569e-02]

  res = optimize.fmin(f, beta0, maxiter=100)
  #res = optimize.fmin(f, np.ones(n_features) / 50., maxiter=10000)
  print res


def svc_cv(X, y, groupsize=200, nfolds=10, C=1.0, kernel='linear'):
  """
  Cross validate a support vector classifier.
  """

  X, y = random_subsets(X, y, n=groupsize)
  folds = StratifiedKFold(y, nfolds)

  for train_indices, test_indices in folds:
    #print train_indices, test_indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    svc = SVC(C=C, kernel=kernel, probability=True, verbose=False)
    svc.fit(X_train, y_train)
    y_prob = svc.predict_proba(X_test)

    print log_loss(y_test, y_prob, svc.classes_)


def svc(n_components=10):
  """
  Train a support vector classifier after dimensionality reduction
  with PCA.

  Each fold takes ~10 min. First fold gave log loss: 0.684875244651
  """

  train = pandas.read_csv('train.csv')
  train = df_random_subsets(train, n=100)
  #prepare(train, log=True)

  #kpca = KernelPCA(kernel='rbf', n_components=n_components, eigen_solver='arpack')
  #kpca.fit(features(train.groupby('target').head(100)))

  y = train['target'].values
  X = np.asarray(1.0 * features(train))
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

    #X_train = kpca.transform(X_train)
    #X_test = kpca.transform(X_test)

    rbf = CustomRBF([
      1.97937904e-02,  2.09697373e-02,  1.97602619e-02,  2.29713609e-02,
      1.89022561e-02,  2.11266579e-02,  2.06388922e-02,  1.97510513e-02,
      1.96269322e-02,  2.01458774e-02,  1.96956710e-02,  2.07461472e-02,
      1.92185214e-02,  1.90597482e-02,  2.09208783e-02,  1.88683077e-02,
      2.05427241e-02,  2.38246369e-02,  1.74301233e-02,  2.01794112e-02,
      2.05120477e-02,  1.99764439e-02,  2.03300554e-02,  1.83119249e-02,
      2.02714147e-02,  2.05032421e-02,  1.94780292e-02,  2.02757550e-02,
      2.00939729e-02,  2.00333158e-02,  2.04026553e-02,  2.08000677e-02,
      2.02023788e-02, -5.90571278e-07,  1.98071893e-02,  2.24698584e-02,
      2.05974852e-02,  1.66059498e-02,  2.24482815e-02,  1.99106743e-02,
      1.83773552e-02,  2.04635946e-02,  1.97173055e-02,  2.05034796e-02,
      2.01263825e-02,  2.05656383e-02,  2.40093408e-02,  1.93932056e-02,
      2.04805362e-02,  2.04506010e-02,  2.06895361e-02,  2.03879553e-02,
      2.06291349e-02,  2.08500963e-02,  1.82011569e-02,  1.95928625e-02,
      2.16621080e-02,  2.00374440e-02,  1.98195672e-02,  1.97160556e-02,
      2.04650218e-02,  2.04733047e-02,  2.00538932e-02,  1.78187375e-02,
      2.08613772e-02,  1.98952890e-02,  1.85132112e-02,  2.06320461e-02,
      1.91273549e-02,  2.05474741e-02,  1.74197293e-02,  1.98496756e-02,
      2.44594952e-02,  1.96383596e-02,  2.01792334e-02,  2.01987450e-02,
      1.97733793e-02,  2.07221204e-02,  2.42708639e-02,  2.04267702e-02,
      2.03453017e-02,  2.02582442e-02,  2.10294289e-02,  1.89568396e-02,
      1.98266333e-02,  2.05653820e-02,  2.07231908e-02,  2.06521860e-02,
      2.08141757e-02,  2.02049611e-02,  2.00691448e-02,  1.94086036e-02,
      1.99469569e-02
    ])
    #rbf = CustomRBF(np.ones(93) / 50.)

    svc = SVC(C=10.0, kernel=rbf, probability=True, verbose=False)
    svc.fit(X_train, y_train)
    y_prob = svc.predict_proba(X_test)

    print log_loss(y_test, y_prob, svc.classes_)



def plot_pca_by_class(prep=True):
  train = pandas.read_csv('train.csv')

  if prep:
    prepare(train)

  i = 1
  pc = np.zeros((9, 93))

  for cls, group in train.groupby('target'):
    pca = PCA()
    pca.fit(1.0 * features(group).values)
    pc[i-1] = pca.components_[0]
    plt.figure(1)
    plt.subplot(3, 3, i)
    plt.plot(pca.explained_variance_ratio_, marker='o')
    plt.title(cls)

    #plt.figure(2)
    #inds = np.argsort(np.abs(pca.components_[0]))
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

  plt.pcolormesh(np.dot(pc, pc.T), vmin=0.0, vmax=1.0)
  plt.title('Orthogonality between class principal components')
  plt.colorbar()
  plt.show()


def cv_pca_by_class(prep=False, nfolds=10):
  train = pandas.read_csv('train.csv')

  if prep:
    prepare(train, log=False)

  classes = ['Class_%d' % i for i in range(1, 10)] #[1, 2, 3, 4]

  y = train['target'].values
  X = np.asarray(1.0 * features(train))
  #X = raw_scaled_features(train, log=True)

  folds = StratifiedKFold(train['target'], nfolds)

  for train_indices, test_indices in folds:
    #print train_indices, test_indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    probs = np.zeros((len(X_test), len(classes)))

    for i, cls in enumerate(classes):
      pca = PCA(n_components=None)
      X_train_cls = X_train[y_train == cls]
      #print cls, 'training on', len(X_train_cls)
      pca.fit(X_train_cls)
      proj = np.dot(X_test, pca.components_.T)
      #proj = np.abs(pca.transform(X_test))
      #print cls, 'scaled projections:', np.abs(proj) * pca.explained_variance_ratio_
      #print cls, pca.explained_variance_ratio_
      # For PCA
      probs[:,i] = np.sum(np.abs(proj) * pca.explained_variance_ratio_, axis=1)
      # For ICA
      #probs[:,i] = np.sum(np.abs(proj), axis=1)

    probs = probs**4
    probs /= np.sum(probs, axis=1).reshape((len(probs), 1))
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


def net():
  train = pandas.read_csv('train.csv')
  Xall = np.log(1.0 + features(train).values)
  #Xall = np.concatenate([Xall, np.ones((len(Xall), 1))], axis=1)
  #Xall = scale(1.0 * features(train).values)
  #Xall = 1.0 * features(train).values
  yall = train['target'].values

  X, y = Xall, yall

  dbn = DBN([X.shape[1], 60, len(np.unique(y))], learn_rates=0.1, epochs=20, verbose=1)
  cv(X, y, [dbn])



#plot_pca(kernel='rbf')
#svc(n_components=20)
#corr()
#pca_by_class()
#cv_pca_by_class(prep=False, nfolds=2)
#plot_pca_by_class(prep=False)
#tmi()
#cluster()
eigencluster2()
#grid()
#knn()
#jumble()
#opt_svc()
#net()
