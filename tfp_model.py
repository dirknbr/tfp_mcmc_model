import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import arviz

tfd = tfp.distributions
tfb = tfp.bijectors

class TFPModel:
  def __init__(self, a_prior=[0, 10], b_prior=[0, 1], b_type='normal', num_chains=2):
    self.a_prior = a_prior # normal
    self.b_prior = b_prior # [1,1] or [1] for halfnormal
    self.b_type = b_type # normal or halfnormal
    self.sigma_prior = [0, .5] # lognormal
    self.num_chains = num_chains
    self.dtype = tf.float32

  def fit(self, X, y):
    X_obs = tf.convert_to_tensor(X, dtype=self.dtype)
    y_obs = tf.convert_to_tensor(y, dtype=self.dtype)
    n, num_feat = X.shape
    def build_target_log_prob_fn(X_obs, y_obs, num_feat):
      # Define prior distributions
      prior_a_dist = tfd.Normal(loc=tf.cast(self.a_prior[0], self.dtype), scale=tf.cast(self.a_prior[1], self.dtype))
      # Prior for b vector (independent normals for each component)
      if self.b_type == 'normal':
        prior_b_dist = tfd.Independent(
            tfd.Normal(loc=tf.ones(num_feat, dtype=self.dtype) * self.b_prior[0], scale=tf.cast(self.b_prior[1], self.dtype)),
            reinterpreted_batch_ndims=1 # Sums log_prob over features
        )
      else:
        prior_b_dist = tfd.Independent(
            tfd.HalfNormal(tf.ones(num_feat, dtype=self.dtype) * self.b_prior[0]),
            reinterpreted_batch_ndims=1 # Sums log_prob over features
        )
      # Prior for log_sigma, so sigma = exp(log_sigma) is positive.
      prior_log_sigma_dist = tfd.Normal(loc=tf.cast(self.sigma_prior[0], self.dtype), scale=tf.cast(self.sigma_prior[1], self.dtype))

      def target_log_prob_fn(a, b, log_sigma):
          # Log probability of priors
          # a and log_sigma have shape [num_chains]
          # b has shape [num_chains, num_feat]
          lp_a = prior_a_dist.log_prob(a)
          lp_b = prior_b_dist.log_prob(b) # This will be [num_chains] due to Independent
          lp_log_sigma = prior_log_sigma_dist.log_prob(log_sigma)
          log_prob_priors = lp_a + lp_b + lp_log_sigma
          
          # Likelihood calculation
          sigma = tf.exp(log_sigma) # Transform log_sigma to sigma > 0
          
          # Ensure correct broadcasting for multiple chains
          # a: [num_chains]
          # b: [num_chains, num_feat]
          # X_obs: [num_datapoints, num_feat]
          # y_obs: [num_datapoints]
          # mu needs to be [num_chains, num_datapoints]
          
          # Calculate X @ b for each chain:
          # tf.einsum('cf,df->cd', b, X_obs) means b[chain, feat] * X_obs[data, feat] summed over feat
          # This results in a shape of [num_chains, num_datapoints]
          mu = a[..., tf.newaxis] + tf.einsum('cf,df->cd', b, X_obs)
          
          likelihood_dist = tfd.Normal(loc=mu, scale=sigma[..., tf.newaxis])
          log_prob_likelihood = tf.reduce_sum(likelihood_dist.log_prob(y_obs), axis=-1)

          return log_prob_priors + log_prob_likelihood
          
      return target_log_prob_fn

    target_log_prob_fn = build_target_log_prob_fn(X_obs, y_obs, num_feat)
    initial_states = [
      tf.zeros(self.num_chains, name='init_a', dtype=self.dtype),
      tf.zeros([self.num_chains, num_feat], name='init_b', dtype=self.dtype), # b is [num_chains, num_feat]
      tf.zeros(self.num_chains, name='init_log_sigma', dtype=self.dtype)
    ]

    nuts_kernel = tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn=target_log_prob_fn,
      step_size=tf.cast(0.1, dtype=self.dtype)
    )

    adaptive_nuts_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
      inner_kernel=nuts_kernel,
      num_adaptation_steps=1000,
      target_accept_prob=tf.cast(0.75, dtype=self.dtype)
    )

    @tf.function(jit_compile=True)
    def run_mcmc_chains():
      samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=1000,
        num_burnin_steps=1000,
        current_state=initial_states,
        kernel=adaptive_nuts_kernel,
        trace_fn=lambda _, pkr: (pkr.inner_results.is_accepted,
                                 pkr.inner_results.log_accept_ratio)
      )
      return samples, kernel_results

    states_list, kernel_results_info = run_mcmc_chains()
    # tfp gives shape (draw, chain, x) and arviz wants (chain, draw, x)
    # https://python.arviz.org/en/latest/getting_started/CreatingInferenceData.html
    self.infdata = arviz.convert_to_inference_data({'a': states_list[0].numpy().T, 
                                                    'b': np.transpose(states_list[1].numpy(), (1, 0, 2)),
                                                    'sigma': tf.exp(states_list[2]).numpy().T})

  def summary(self):
    return arviz.summary(self.infdata)

  def predict(self, X):
    # make mean prediction
    a = np.array(self.infdata.posterior['a'].mean())
    b = np.array(self.infdata.posterior['b'].mean(axis=(0, 1)))
    return a + X.dot(b)






    

    