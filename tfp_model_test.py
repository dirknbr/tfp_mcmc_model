from tfp_model import *
import unittest

class ModelTests(unittest.TestCase):
  def test_fit_and_predict(self):
    n = 100
    X = np.random.normal(0, 1, (n, 3))
    y = 10 + X[:, 1] + X[:, 2] * .5 + np.random.normal(0, 1, n)
    model = TFPModel()
    model.fit(X, y)
    print(model.summary())
    pred = model.predict(X)
    self.assertEqual(len(pred), len(y))
    self.assertEqual(model.infdata.posterior['b'].shape, (2, 1000, 3)) # (chain, draw, x)

  def test_half_normal(self):
    n = 100
    X = np.random.normal(0, 1, (n, 3))
    y = 10 + X[:, 1] + X[:, 2] * .5 + np.random.normal(0, 1, n)
    model = TFPModel(b_type='halfnormal', b_prior=[1])
    model.fit(X, y)
    print(model.summary())
    pred = model.predict(X)
    self.assertEqual(len(pred), len(y))

if __name__ == '__main__':
  unittest.main()