data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1,upper=J> country[N];
  int<lower=0> new_cases[N];
  vector<lower=0>[N] cases;
  vector<lower=1>[N] days;
  int M;
  matrix[N,M] Measures;
  int W;
  matrix[N,W] Weekdays;
}


parameters {
  real alpha;
  real beta;
  vector[J] tau;
  vector[W] gamma;
  vector<upper=0>[M] theta;
  real<lower=0> sd_tau;
  real<lower=0> xi;
}


transformed parameters {
  vector[N] eta = log(cases) + alpha + tau[country] + beta * days + Weekdays * gamma + Measures * theta;
  vector[N] mu = exp(eta);
  real<lower=0> phi = (1. / xi) ^ 2;
}


model { 
  alpha ~ student_t(7., 0., 10.0);
  beta ~ student_t(7., 0., 2.5);
  tau ~ normal(0., sd_tau); 
  sd_tau ~ student_t(4., 0., 1.);
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
