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
initialize_modeling_system_pro()

## -----------------------------------------------------------------------------
# Run a subset of available prognostic models. If all, use model = "all_pro".
results_all_pro <- models_pro(train_pro, model = c("lasso_pro", "rsf_pro"))

# Print summary for Random Survival Forest
print_model_summary_pro("rsf_pro", results_all_pro$rsf_pro)

## -----------------------------------------------------------------------------
# Create a Bagging ensemble with lasso as the base survival model
# n_estimators is reduced for faster execution.
bagging_lasso_pro_results <- bagging_pro(train_pro, base_model_name = "lasso_pro", n_estimators = 5, seed = 123)
print_model_summary_pro("Bagging (LASSO)", bagging_lasso_pro_results)

## -----------------------------------------------------------------------------
# Create a Stacking ensemble with lasso as the meta-model
stacking_lasso_pro_results <- stacking_pro(
  results_all_models = results_all_pro,
  data = train_pro,
  meta_model_name = "lasso_pro"
)
print_model_summary_pro("Stacking (LASSO)", stacking_lasso_pro_results)

## -----------------------------------------------------------------------------
# Apply the trained stacking model to the test set
pro_pred_new <- apply_pro(
  trained_model_object = stacking_lasso_pro_results$model_object,
  new_data = test_pro,
  time_unit = "day"
)

# Evaluate the new prognostic scores
eval_pro_new <- evaluate_predictions_pro(
  prediction_df = pro_pred_new,
  years_to_evaluate = c(1,3, 5)
)
print(eval_pro_new)

## ----fig.width=6, fig.height=5, warning=FALSE---------------------------------
# Kaplan-Meier Curve
p4 <- figure_pro(type = "km", data = stacking_lasso_pro_results, time_unit= "days")
#print(p4)

# Time-Dependent ROC Curve
p5 <- figure_pro(type = "tdroc", data = stacking_lasso_pro_results, time_unit = "days")
#print(p5)

## ----include=FALSE------------------------------------------------------------
# Stop the parallel cluster
parallel::stopCluster(cl)

