
setwd("D:/Chess/Old Indian")

# Simulate chess ratings and outcomes based on group-level summary stats
library(tidyverse)
# Simulate chess ratings and outcomes based on group-level summary stats
library(dplyr)
library(purrr)
library(tibble)
library(tidyr)
library(lme4)
library(e1071)  # for skewness and kurtosis
library(Hmisc)
library(binom)


# Load your dataset
chess_data <- read.csv("data_for_simulation.csv")

# Rename for clarity
chess_data <- chess_data %>%
  rename(
    white_rating = opp,
    black_rating = rating
  )

# Compute win/lose proportions
chess_data <- chess_data %>%
  mutate(
    win_prop = win / (win + lose),
    lose_prop = lose / (win + lose)
  )

# Get minimum non-zero lose proportion (for gid 2 override)
min_lose_prop <- chess_data %>%
  filter(lose_prop > 0, gid != 2) %>%
  summarise(min_lose = min(lose_prop)) %>%
  pull(min_lose)

# Get average win/lose proportions (for gid 1 fallback)
avg_props <- chess_data %>%
  filter(gid != 1) %>%
  summarise(
    avg_win = mean(win_prop),
    avg_lose = mean(lose_prop)
  )

# Get pooled mean and SD across gids (for gid 1 distribution override)
pooled_mu <- chess_data %>%
  filter(gid != 1) %>%
  summarise(mu = mean(avgrating)) %>%
  pull(mu)

pooled_sd <- chess_data %>%
  filter(gid != 1) %>%
  summarise(sd = mean(avgstd)) %>%
  pull(sd)

# Simulation function per group and replicate
simulate_group <- function(group_data, sim_group) {
  gid_val <- unique(group_data$gid)
  n_sim <- 1000
  
  # Override mu/sigma for gid 1 to avoid degenerate distribution
  mu <- if (gid_val == 1) pooled_mu else unique(group_data$avgrating)
  sigma <- if (gid_val == 1) pooled_sd else unique(group_data$avgstd)
  
  # Determine win probability from group-level summary
  win_p <- if (gid_val == 2) {
    1 - min_lose_prop  # override all-win group
  } else if (gid_val == 1) {
    avg_props$avg_win
  } else {
    unique(group_data$win_prop)
  }
  
  # Simulate ratings (rounded to integers)
  black_rating <- round(rnorm(n_sim, mean = mu, sd = sigma))
  white_rating <- round(rnorm(n_sim, mean = mu, sd = sigma))
  
  # Simulate outcomes
  result <- rbinom(n_sim, size = 1, prob = win_p)
  elo_prob <- 1 / (1 + 10 ^ ((white_rating - black_rating) / 400))
  elo_result <- rbinom(n_sim, size = 1, prob = elo_prob)
  
  tibble(
    gid = gid_val,
    sim_group = sim_group,
    black_rating = black_rating,
    white_rating = white_rating,
    result = result,
    elo_prob = elo_prob,
    elo_result = elo_result
  )
}

# Perform 10 replicates per gid
grouped_data <- split(chess_data, chess_data$gid)

simulated_data <- map_dfr(names(grouped_data), function(gid_val) {
  group_df <- grouped_data[[gid_val]]
  map_dfr(1:10, ~simulate_group(group_df, .x))
})

# Preview the result
head(simulated_data)

model <- glm(result ~ black_rating, data = simulated_data, family = binomial)
summary(model)

# Create a sequence of rating values for prediction
new_data <- tibble(black_rating = seq(min(simulated_data$black_rating),
                                      max(simulated_data$black_rating), length.out = 300))

# Get predicted probabilities and confidence intervals
pred <- predict(model, newdata = new_data, type = "link", se.fit = TRUE)

# Compute confidence bands in log-odds scale and convert to probabilities
new_data <- new_data %>%
  mutate(
    fit = pred$fit,
    se = pred$se.fit,
    lower = fit - 1.96 * se,
    upper = fit + 1.96 * se,
    prob = plogis(fit),
    prob_lower = plogis(lower),
    prob_upper = plogis(upper)
  )

library(ggplot2)

ggplot(new_data, aes(x = black_rating, y = prob)) +
  geom_line(color = "blue", size = 1) +
  geom_ribbon(aes(ymin = prob_lower, ymax = prob_upper), alpha = 0.2, fill = "blue") +
  labs(
    title = "Probability of Win by Black vs Black Rating",
    x = "Black Rating",
    y = "P(Win by Black)"
  ) +
  theme_minimal()

ggplot(new_data, aes(x = black_rating, y = prob)) +
  # Jittered binary outcomes
  geom_jitter(data = simulated_data, 
              aes(x = black_rating, y = result), 
              height = 0.05, width = 0, 
              alpha = 0.3, color = "gray30") +
  # Fitted sigmoid curve
  geom_line(color = "blue", size = 1) +
  # Confidence interval ribbon
  geom_ribbon(aes(ymin = prob_lower, ymax = prob_upper), alpha = 0.2, fill = "blue") +
  labs(
    title = "Probability of Win by Black vs Black Rating",
    x = "Black Rating",
    y = "P(Win by Black)"
  ) +
  theme_minimal()

library(tidyverse)
library(broom)

# Fit logistic regression within each group
models_by_gid <- simulated_data %>%
  group_by(gid) %>%
  nest() %>%
  mutate(
    model = map(data, ~ glm(result ~ black_rating, data = ., family = binomial)),
    new_data = map(data, ~ tibble(black_rating = seq(min(.$black_rating),
                                                     max(.$black_rating), length.out = 200))),
    pred = map2(model, new_data, ~ {
      p <- predict(.x, newdata = .y, type = "link", se.fit = TRUE)
      .y %>%
        mutate(
          fit = p$fit,
          se = p$se.fit,
          lower = fit - 1.96 * se,
          upper = fit + 1.96 * se,
          prob = plogis(fit),
          prob_lower = plogis(lower),
          prob_upper = plogis(upper)
        )
    })
  )

pred_data <- models_by_gid %>%
  select(gid, pred) %>%
  unnest(pred)

ggplot(pred_data, aes(x = black_rating, y = prob)) +
  geom_jitter(data = simulated_data,
              aes(x = black_rating, y = result),
              height = 0.05, width = 0, alpha = 0.3, color = "gray30") +
  geom_line(color = "blue", size = 1) +
  geom_ribbon(aes(ymin = prob_lower, ymax = prob_upper), alpha = 0.2, fill = "blue") +
  facet_wrap(~ gid) +
  labs(
    title = "Logistic Regression by Group",
    x = "Black Rating",
    y = "P(Win by Black)"
  ) +
  theme_minimal()

pooled_model <- glm(result ~ black_rating, data = simulated_data, family = binomial)
rating_grid <- tibble(black_rating = seq(min(simulated_data$black_rating),
                                         max(simulated_data$black_rating), length.out = 300))

pooled_pred <- predict(pooled_model, newdata = rating_grid, type = "link", se.fit = TRUE)

rating_grid <- rating_grid %>%
  mutate(
    fit = pooled_pred$fit,
    se = pooled_pred$se.fit,
    lower = fit - 1.96 * se,
    upper = fit + 1.96 * se,
    prob = plogis(fit),
    prob_lower = plogis(lower),
    prob_upper = plogis(upper)
  )

ggplot(simulated_data, aes(x = black_rating, y = result)) +
  # Jittered points from original data
  geom_jitter(height = 0.05, width = 0, alpha = 0.3, color = "gray40") +
  
  # Pooled sigmoid curve (must override global aes)
  geom_line(
    data = rating_grid,
    aes(x = black_rating, y = prob),
    inherit.aes = FALSE,
    color = "blue", size = 1
  ) +
  
  # Confidence interval ribbon (also override global aes)
  geom_ribbon(
    data = rating_grid,
    aes(x = black_rating, ymin = prob_lower, ymax = prob_upper),
    inherit.aes = FALSE,
    fill = "blue", alpha = 0.2
  ) +
  
  facet_wrap(~ gid) +
  labs(
    title = "Pooled Logistic Regression with Faceted Raw Data",
    x = "Black Rating",
    y = "P(Win by Black)"
  ) +
  theme_minimal()




# Optionally write to file
write.csv(simulated_data, "simulated_chess_data.csv", row.names = FALSE)

library(ggplot2)

# Density plot of black_rating by gid
ggplot(simulated_data, aes(x = black_rating, color = factor(gid), fill = factor(gid))) +
  geom_density(alpha = 0.3) +
  labs(
    title = "Density of Simulated Black Ratings by GID",
    x = "Simulated Black Rating",
    y = "Density",
    color = "GID",
    fill = "GID"
  ) +
  theme_minimal()


simulated_data %>%
  filter(gid == 1) %>%
  summarise(
    skew = skewness(black_rating),
    kurt = kurtosis(black_rating),
    min = min(black_rating),
    max = max(black_rating),
    sd = sd(black_rating)
  )

# Convert from wide to long format
long_data <- simulated_data %>%
  pivot_longer(
    cols = c(result, elo_result),
    names_to = "method",
    values_to = "outcome"
  )
head(long_data)

# ANOVA: does outcome differ by method and gid?
anova_model <- aov(outcome ~ method * factor(gid), data = long_data)
summary(anova_model)
TukeyHSD(anova_model)


# Reshape to long format
long_data <- simulated_data %>%
  pivot_longer(
    cols = c(result, elo_result),
    names_to = "method",
    values_to = "outcome"
  )
head(long_data)
# Convert method to factor
long_data$method <- factor(long_data$method, levels = c("result", "elo_result"))

# Fit mixed model with random intercept for gid
model <- glmer(outcome ~ method + (1 | gid), data = long_data, family = binomial)

summary(model)

# Extract fixed effects
fixef_vals <- fixef(model)

# Baseline (method = "result") log-odds
logodds_result <- fixef_vals[["(Intercept)"]]

# Method difference: log-odds for elo_result - result
logodds_diff <- fixef_vals[["methodelo_result"]]

# Convert to probabilities
prob_result <- plogis(logodds_result)
prob_elo_result <- plogis(logodds_result + logodds_diff)

# Print
cat("Estimated P(success) for 'result':", round(prob_result, 3), "\n")
cat("Estimated P(success) for 'elo_result':", round(prob_elo_result, 3), "\n")


# Get predicted probabilities per gid
group_preds <- long_data %>%
  group_by(gid, method) %>%
  summarise(
    mean_outcome = mean(outcome),
    .groups = "drop"
  )

# Plot
ggplot(group_preds, aes(x = factor(gid), y = mean_outcome, fill = method)) +
  geom_col(position = position_dodge()) +
  labs(
    title = "Average Simulated Success Rate by Method and GID",
    x = "GID",
    y = "Mean Success Probability",
    fill = "Method"
  ) +
  theme_minimal()

# Reshape to long format
long_data <- simulated_data %>%
  pivot_longer(
    cols = c(result, elo_result),
    names_to = "method",
    values_to = "outcome"
  ) %>%
  mutate(method = factor(method, levels = c("result", "elo_result")))


# Compute group-wise success rates and Wilson CIs, avoiding column conflicts
group_preds_ci <- long_data %>%
  group_by(gid, method) %>%
  summarise(
    successes = sum(outcome),
    trials = n(),
    .groups = "drop"
  ) %>%
  mutate(
    binom_stats = map2(successes, trials, ~ binom::binom.confint(x = .x, n = .y, methods = "wilson") %>%
                         select(mean = mean, lower = lower, upper = upper))
  ) %>%
  unnest(binom_stats)



# Plot with CIs

ggplot(group_preds_ci, aes(x = factor(gid), y = mean, fill = method)) +
  geom_col(position = position_dodge(width = 0.9), width = 0.7) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    position = position_dodge(width = 0.9),
    width = 0.2
  ) +
  labs(
    title = "Simulated Win Rates by Method and GID with 95% CI",
    x = "GID",
    y = "Estimated Win Probability",
    fill = "Method"
  ) +
  theme_minimal()






























model_nested <- glmer(outcome ~ method + (1 | gid/sim_group), data = long_data, family = binomial)
summary(model_nested)

