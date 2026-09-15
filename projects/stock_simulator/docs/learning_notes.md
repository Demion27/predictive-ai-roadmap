# Brownian Motion
Brownian motion is a continuous-time stochastic process, often denoted as a Wiener process ($W_{t}$), defined by random, jagged, and non-differentiable paths with stationary, independent Gaussian increments. It represents the mathematical limit of a scaled random walk, starting at $(W_{0}=0)$, with variance $(W_{t}=t)$. 
Brownian motion is used in financial mathematics to model random, unpredictable movements in asset prices and interest rates. 
## Mathematical Definition
A standard Brownian motion $(W_{t})$ has $(W_{0}=0)$, independent increments $(W_{t}-W_{s}$~ N(0,t-s) for (0<s<t), and continuous paths.
# Geometric Brownian Motion (GBM)
Stock prices are often modeled using GBM to ensure they remain positive, defined as $$dS_{t}=μ S_{t}dt+σ S_{t}dW_{t}\)$$, where μ is drift and σ is volatility both derived from the mean and standard deviation of log returns respectively. This project will use GBM to keep stock prices positive.
Using itos lemma future random stock prices can be calculated as $S_{t} = S_{o}\exp{(μ-\frac{σ}{2}+σW)}$
## Additional Information
The wiener process at any interval is centered at the beginning of the interval. It is used to introduce randomness to the stock prices as it does not depend on previous information.
μ (drift) affects the mean of the distribution as the mean shifts by μt 
σ (volatility) determines the magnitude of the fluctuations
t (time) affects variance which means as time moves forward the ditribution grows wider