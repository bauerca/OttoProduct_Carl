# Kaggle: Otto Group Classification Challenge

- [Competition homepage](https://www.kaggle.com/c/otto-group-product-classification-challenge)
- [Scoring](https://www.kaggle.com/c/otto-group-product-classification-challenge/details/evaluation)

## Usage

A bunch of functions are defined within `otto.py`. Invoke them from the
bottom of the script (by uncommenting them) and run

```
python otto.py
```

You're gonna need the following python packages installed:

- [scikit learn](http://scikit-learn.org/stable/)
- [pandas](http://pandas.pydata.org/)
- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)

### Principal component analysis

```python
plot_pca()
```

This function will run a principal component analysis (PCA) on the full training
dataset (where features have been scaled and shifted to zero mean and unit variance)
and plot:

- The variances for all the principal components, and
- Projections onto 2D PCA component subspaces (the subspaces with the most variance)
  of Otto Products from pairwise classes


### Support vector machine

```python
svc(n_components=20)
```

Run a support vector classification on `n_components` from a PCA-based dimensionality
reduction.

Some results

`n_components`|log loss| runtime |
--------------|--------|---------|
  10          |  0.685 | 10 min  |


## License

MIT
