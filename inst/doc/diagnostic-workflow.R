## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(E2E)

## ----include=FALSE------------------------------------------------------------
# Set up a 2-core cluster for parallel processing in this vignette
# This is crucial for passing R CMD check on CI/CD platforms
cl <- parallel::makeCluster(2)
doParallel::registerDoParallel(cl)

## -----------------------------------------------------------------------------
initialize_modeling_system_dia()

## -----------------------------------------------------------------------------
# To run all, use model = "all_dia" or omit the parameter.
results_all_dia <- models_dia(train_dia, model = c("rf", "lasso", "xb"))

# Print a summary for a specific model (e.g., Random Forest)
print_model_summary_dia("rf", results_all_dia$rf)

## -----------------------------------------------------------------------------
# Run a specific subset of models with tuning enabled and custom thresholds
results_dia_custom <- models_dia(
  data = train_dia,
  model = c("rf", "lasso", "xb"),
  tune = TRUE,
  seed = 123,
  threshold_choices = list(rf = "f1", lasso = 0.6, xb = "youden"),
  positive_label_value = 1,
  negative_label_value = 0,
  new_positive_label = "Case",
  new_negative_label = "Control"
)

# View the custom results
print_model_summary_dia("rf", results_dia_custom$rf)

## -----------------------------------------------------------------------------
# Create a Bagging ensemble with XGBoost as the base model
# n_estimators is reduced for faster execution in this example.
bagging_xb_results <- bagging_dia(train_dia, base_model_name = "xb", n_estimators = 5)
print_model_summary_dia("Bagging (XGBoost)", bagging_xb_results)

## -----------------------------------------------------------------------------
# Create a soft voting ensemble from the top models
voting_soft_results <- voting_dia(
  results_all_models = results_all_dia,
  data = train_dia,
  type = "soft"
)
print_model_summary_dia("Voting (Soft)", voting_soft_results)

## -----------------------------------------------------------------------------
# Create a Stacking ensemble with Lasso as the meta-model
stacking_lasso_results <- stacking_dia(
  results_all_models = results_all_dia,
  data = train_dia,
  meta_model_name = "lasso"
)
print_model_summary_dia("Stacking (Lasso)", stacking_lasso_results)

## -----------------------------------------------------------------------------
# Create an EasyEnsemble with XGBoost as the base model
# n_estimators is reduced for faster execution.
results_imbalance_dia <- imbalance_dia(train_dia, base_model_name = "xb", n_estimators = 5, seed = 123)
print_model_summary_dia("Imbalance (XGBoost)", results_imbalance_dia)

## -----------------------------------------------------------------------------
# Apply the trained Bagging model to the test set
bagging_pred_new <- apply_dia(
  trained_model_object = bagging_xb_results$model_object,
  new_data = test_dia,
  label_col_name = "outcome",
  pos_class = "Positive",
  neg_class = "Negative"
)

# Evaluate these new predictions
eval_results_new <- evaluate_model_dia(
  precomputed_prob = bagging_pred_new$score,
  y_data = factor(test_dia$outcome, levels = c(0, 1), labels = c("Positive", "Negative")),
  sample_ids = test_dia$sample,
  threshold_strategy = "default",
  pos_class = "Positive",
  neg_class = "Negative"
)
print(eval_results_new$evaluation_metrics)

## ----fig.width=5, fig.height=5, warning=FALSE---------------------------------
# ROC Curve
p1 <- figure_dia(type = "roc", data = results_imbalance_dia)
#plot(p1)

# Precision-Recall Curve
p2 <- figure_dia(type = "prc", data = results_imbalance_dia)
#plot(p2)

# Confusion Matrix
p3 <- figure_dia(type = "matrix", data = results_imbalance_dia)
#plot(p3)

## ----include=FALSE------------------------------------------------------------
# Stop the parallel cluster
parallel::stopCluster(cl)

