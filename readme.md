# Simple Bayesian regression in TFP

The goal here is to hide the TFP MCMC complexity and make this run really fast.

The model is $y \sim N(a + b * X, \sigma)$ where a and b have normal priors that can be changed.

```
    n = 100
    X = np.random.normal(0, 1, (n, 3))
    y = 10 + X[:, 1] + X[:, 2] * .5 + np.random.normal(0, 1, n)
    model = TFPModel()
    model.fit(X, y)
    print(model.summary())
    pred = model.predict(X)
```
