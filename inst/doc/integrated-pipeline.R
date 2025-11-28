## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 10,
  fig.height = 8
)

## ----setup--------------------------------------------------------------------
library(E2E)

## ----include=FALSE------------------------------------------------------------
# Set up parallel processing
cl <- parallel::makeCluster(2)
doParallel::registerDoParallel(cl)

## ----eval=FALSE---------------------------------------------------------------
# # Run all diagnostic models
# results_dia <- int_dia(
#   train_dia,
#   test_dia,
#   test_dia, #can be any other data
#   tune = TRUE,
#   n_estimators = 5,
#   seed = 123
# )
# 
# # Visualize results
# #plot_integrated_results(results_dia, metric_name = "AUROC")

## ----eval=FALSE---------------------------------------------------------------
# # Run all models including imbalance handling methods
# results_imb <- int_imbalance(
#   train_dia,
#   test_dia,
#   test_dia, #can be any other data
#   tune = TRUE,
#   n_estimators = 5,
#   seed = 123
# )
# 
# # Visualize results
# #plot_integrated_results(results_imb, metric_name = "AUROC")

## ----eval=FALSE---------------------------------------------------------------
# # Run all prognostic models
# results_pro <- int_pro(
#   train_pro,
#   test_pro,
#   test_pro, #can be any other data
#   tune = TRUE,
#   n_estimators = 5,
#   time_unit = "day",
#   years_to_evaluate = c(1, 3, 5),
#   seed = 123
# )
# 
# # Visualize results (C-index)
# #plot_integrated_results(results_pro, metric_name = "C-index")

## ----include=FALSE------------------------------------------------------------
# Stop parallel cluster
parallel::stopCluster(cl)

