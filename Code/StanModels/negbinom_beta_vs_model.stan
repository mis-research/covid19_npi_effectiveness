data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1,upper=J> country[N];
  int<lower=0> new_cases[N];
  vector<lower=0>[N] cases;
  int<lower=0> days[N];
  int M;
  matrix[N,M] Measures;
  int W;
  matrix[N,W] Weekdays;
}


parameters {
  real alpha;
  real beta;
  matrix[2,J] tau_c;
  cholesky_factor_corr[2] L_c;
  vector<lower=0>[2] sd_c;
  vector[W] gamma;
  vector<upper=0>[M] theta;
  real<lower=0> xi;
} 


transformed parameters {
  matrix[2,J] tau = diag_pre_multiply(sd_c, L_c) * tau_c;
  real phi = (1. / xi) ^ 2; 
  vector[N] eta;
  vector[N] mu;
  for (i in 1:N) {
    eta[i] = log(cases[i]) + alpha + tau[1,country[i]] + beta * days[i] + tau[2,country[i]] * days[i] + Weekdays[i,1:W] * gamma + Measures[i,1:M] * theta;
  }
  mu = exp(eta);
} 


model {
  alpha ~ student_t(7., 0., 10.0);
  beta ~ student_t(7., 0., 2.5);
  to_vector(tau_c) ~ normal(0., 1.);
  L_c ~ lkj_corr_cholesky(1.);
  sd_c ~ student_t(4, 0., 1.);
  gamma ~ student_t(7., 0., 2.5); 
  theta ~ student_t(7., 0., 2.5);
  xi ~ normal(0., 1.);
  for (i in 1:N) {
    new_cases[i] ~ neg_binomial_2_log(eta[i], phi);
  }
}


generated quantities {
  vector[N] log_lik;
  //vector[N] new_cases_rep;
  for (i in 1:N) {
    log_lik[i] = neg_binomial_2_log_lpmf(new_cases[i] | eta[i], phi);
    //new_cases_rep[i] = neg_binomial_2_rng(mu[i], phi);
  }
}
