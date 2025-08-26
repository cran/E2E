## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(E2E)
# Initialize systems for the examples
initialize_modeling_system_dia()
initialize_modeling_system_pro()

## ----include=FALSE------------------------------------------------------------
# Set up a 2-core cluster for parallel processing in this vignette
# This is crucial for passing R CMD check on CI/CD platforms
cl <- parallel::makeCluster(2)
doParallel::registerDoParallel(cl)

## -----------------------------------------------------------------------------
# 1. Define the model function (must accept X, y, and other standard args)
ab_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  # Ensure caret is available
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("Package 'caret' is required for this custom model.")
  }
  ctrl <- caret::trainControl(method = "cv", number = cv_folds,
                              classProbs = TRUE, summaryFunction = caret::twoClassSummary)
  grid <- if (tune) {
    expand.grid(iter = c(50, 100), maxdepth = c(1, 2), nu = 0.1)
  } else {
    expand.grid(iter = 50, maxdepth = 1, nu = 0.1)
  }
  caret::train(x = X, y = y, method = "ada", metric = "ROC", trControl = ctrl, tuneGrid = grid)
}

# 2. Register the model with a unique name
register_model_dia("ab", ab_dia)

# 3. Now you can use "ab" in any diagnostic function
results_ab <- models_dia(train_dia, model = "ab")
print_model_summary_dia("ab", results_ab$ab)

## ----fig.width=7, fig.height=6, warning=FALSE---------------------------------
# First, we need a model result object
bagging_xb_results <- bagging_dia(train_dia, base_model_name = "xb", n_estimators = 10, seed=123)

# Now, generate the SHAP explanation plot
p6 <- figure_shap(
  data = bagging_xb_results,
  raw_data = train_dia,
  target_type = "diagnosis"
)
#plot(p6)

## ----fig.width=7, fig.height=6, warning=FALSE---------------------------------
# First, we need a model result object
stacking_stepcox_pro_results <- stacking_pro(
  results_all_models = models_pro(train_pro, model = c("lasso_pro", "rsf_pro")),
  data = train_pro,
  meta_model_name = "stepcox_pro"
)

# Generate the SHAP explanation plot
p7 <- figure_shap(
  data = stacking_stepcox_pro_results,
  raw_data = train_pro,
  target_type = "prognosis"
)
#plot(p7)

## ----include=FALSE------------------------------------------------------------
# Stop the parallel cluster
parallel::stopCluster(cl)

