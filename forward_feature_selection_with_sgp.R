pdist <- function(X1, X2) {
  if (identical(X1, X2) == TRUE) {
    D <- as.matrix(dist(X1))
  }
  else {
    D <- as.matrix(dist(rbind(X1, X2)))
    D <- D[1:nrow(X1), (nrow(X1) + 1):(nrow(X1) + nrow(X2))]
  }
  return(D)
}

set.seed(1)
source(file = 'sgp_train.R')
source(file = 'sgp_test.R')

locations <-  c('ny', 'nj', 'il', 'ca', 'ma')
#locations <- c('new york', 'massachusetts', 'new jersey', 'california', 'illinois')

response <- 'cases'
unit <- '7d_moving_average_'
load(sprintf('./data/response_data/%scdc_%s.RData', unit, response))
start_date_training <- which(colnames(case_counts) == '4/12/20') # Sunday
training_sizes <- seq(start_date_training, ncol(case_counts) + 1, 7)

for(train_size in training_sizes){
  for(location in locations){
    # load data
    load(sprintf('./data/response_data/%scdc_%s.RData', unit, response))
    load('./data/spatial_data/X_spaial.RData')
    
    X_spatial_all <- X_spatial[, c('n_pop_2018', 'n_people_below_poverty_level_2017', 'avg_hh_size_2010', 'n_households_2010')]
    X_spatial_all[, 'n_nonwhite'] <- X_spatial[, 'p_nonwhite'] * X_spatial[, 'n_pop_2018']
    X_spatial_all[, 'n_20_65_age'] <- rowSums(X_spatial[, c('age20', 'age25', 'age30', 'age35', 'age40', 'age45', 'age50', 'age55', 'age60')])
    
    load('../data/spatial_data/county_coordinates.RData')
    load('../data/response_data/population.RData')
    
    # use the counties from spatial feature matrix X_spatial
    counties <- rownames(X_spatial)
    case_counts <- case_counts[counties, 1:train_size]
    
    # temporal features
    X_temporal <- as.data.frame(matrix(NA, nrow = train_size, ncol = 2), row.names = colnames(case_counts))
    colnames(X_temporal) <- c('week', 'day')
    weeks <- rep(4:52, each = 7)[-c(1:4)] 
    X_temporal[,'week'] <- weeks[1:train_size]
    X_temporal[,'day'] <- 1:train_size
    
    # select counties of a state given its fips code range
    if(location == locations[1]) {selected_county <- which(as.numeric(rownames(county_coordinates)) < 37000 & as.numeric(rownames(county_coordinates)) > 36000)} #NY
    if(location == locations[2]) {selected_county <- which(as.numeric(rownames(county_coordinates)) < 35000 & as.numeric(rownames(county_coordinates)) > 34000)} #NJ
    if(location == locations[3]) {selected_county <- which(as.numeric(rownames(county_coordinates)) < 18000 & as.numeric(rownames(county_coordinates)) > 17000)} #IL
    if(location == locations[4]) {selected_county <- which(as.numeric(rownames(county_coordinates)) < 07000 & as.numeric(rownames(county_coordinates)) > 06000)} #CA
    if(location == locations[5]) {selected_county <- which(as.numeric(rownames(county_coordinates)) < 26000 & as.numeric(rownames(county_coordinates)) > 25000)} #MA
    
    cases <- case_counts[selected_county, ]
    coords <- county_coordinates[selected_county, ]
    
    forecast_days <- 7
    spatial_train_indices <- 1:nrow(cases)
    spatial_test_indices <- 1:nrow(cases)
    temporal_val_train_indices <- 1:(train_size - (2 * forecast_days))
    temporal_val_test_indices <- (train_size - (2 * forecast_days) + 1):(train_size - forecast_days)
    
    # responce data
    Y_val_tra <- log2(as.matrix(cases[spatial_train_indices, temporal_val_train_indices]) + 1)
    Y_val_test <- log2(as.matrix(cases[spatial_test_indices, temporal_val_test_indices]) + 1)
    
    s_ratio_temporal <- sqrt(0.5)
    s_ratio_spatial <- sqrt(0.5)
    
    # temporal data
    X_temporal_val_tra <- as.matrix(scale(X_temporal[temporal_val_train_indices,]))
    X_temporal_val_test <- as.matrix(X_temporal[temporal_val_test_indices,])
    X_temporal_val_test <- (X_temporal_val_test - matrix(attr(X_temporal_val_tra, "scaled:center"), nrow = nrow(X_temporal_val_test), ncol = ncol(X_temporal_val_test), byrow = TRUE)) / matrix(attr(X_temporal_val_tra, "scaled:scale"), nrow = nrow(X_temporal_val_test), ncol = ncol(X_temporal_val_test), byrow = TRUE)
    
    temporal_features <- ncol(X_temporal)
    
    K_temporal_tra_tra <- matrix(1, nrow = nrow(X_temporal_val_tra), ncol = nrow(X_temporal_val_tra))
    K_temporal_test_tra <- matrix(1, nrow = nrow(X_temporal_val_test), ncol = nrow(X_temporal_val_tra))
    K_temporal_test_test <- matrix(1, nrow = nrow(X_temporal_val_test), ncol = nrow(X_temporal_val_test))
    
    for(feature in 1:temporal_features){
      K_temporal_tra_tra <- K_temporal_tra_tra + exp(-(pdist(X_temporal_val_tra[,feature, drop = FALSE], X_temporal_val_tra[,feature, drop = FALSE]))^2 / (2 * s_ratio_temporal^2))
      K_temporal_test_tra <- K_temporal_test_tra + exp(-(pdist(X_temporal_val_test[,feature, drop = FALSE], X_temporal_val_tra[,feature, drop = FALSE]))^2 / (2 * s_ratio_temporal^2))
      K_temporal_test_test <- K_temporal_test_test + exp(-(pdist(X_temporal_val_test[,feature, drop = FALSE], X_temporal_val_test[,feature, drop = FALSE]))^2 / (2 * s_ratio_temporal^2))
    }
    
    sigma_set <- c(1/8, 1/4, 1/2, 1, 2, 4, 8) * mean(apply(Y_val_tra, 1, var))
    cv_selected_features <- list()
    nrmse_matrix <- matrix(NA, nrow = 1, ncol = length(sigma_set), dimnames = list('nrmse', sprintf("%g", sigma_set)))
    for(sigma in sigma_set){
      parameters <- list(sigma = sigma)
      all_features <- colnames(X_spatial_all)
      best_result <- c()
      add_feature <- c()
      save_best_results <- c() 
      load('../data/spatial_data/county_coordinates.RData')
      coords <- county_coordinates[selected_county, ]
      
      while (TRUE) {
        X_spatial <- c()
        results_table <- c()
        for(feature in all_features){
          X_spatial <- cbind(coords, X_spatial_all[selected_county, feature])
          
          X_spatial_tra <- X_spatial[spatial_train_indices, ]
          rownames(X_spatial_tra) <- rownames(X_spatial)[spatial_train_indices]
          X_spatial_test <- X_spatial[spatial_test_indices, ]
          rownames(X_spatial_test) <- rownames(X_spatial)[spatial_test_indices]
          
          X_spatial_tra <- scale(X_spatial_tra)
          X_spatial_test <- (X_spatial_test - matrix(attr(X_spatial_tra, "scaled:center"), nrow = nrow(X_spatial_test), ncol = ncol(X_spatial_test), byrow = TRUE)) / matrix(attr(X_spatial_tra, "scaled:scale"), nrow = nrow(X_spatial_test), ncol = ncol(X_spatial_test), byrow = TRUE)
          
          D_latlong_tra_tra <- pdist(X_spatial_tra[,1:2, drop = FALSE], X_spatial_tra[,1:2, drop = FALSE])
          D_latlong_test_tra <- pdist(X_spatial_test[,1:2, drop = FALSE], X_spatial_tra[,1:2, drop = FALSE])
          D_latlong_test_test <- pdist(X_spatial_test[,1:2, drop = FALSE], X_spatial_test[,1:2, drop = FALSE])
          s <- s_ratio_spatial * mean(D_latlong_tra_tra)
          K_spatial_tra_tra <- exp(-D_latlong_tra_tra^2 / (2 * s^2))
          K_spatial_test_tra <- exp(-D_latlong_test_tra^2 / (2 * s^2))
          K_spatial_test_test <- exp(-D_latlong_test_test^2 / (2 * s^2))
          
          if(length(best_result) == 0){
            state <- structured_gpr_train(1/(ncol(X_spatial) - 1) * K_spatial_tra_tra, 0.5 * K_temporal_tra_tra, Y_val_tra, parameters)
            prediction <- structured_gpr_test(1/(ncol(X_spatial) - 1) * K_spatial_test_tra, 1/(ncol(X_spatial) - 1) * K_spatial_test_test, 0.5 * K_temporal_test_tra, 0.5 * K_temporal_test_test, state)
            best_result <- sqrt(mean(((2^Y_val_test - 1) - round(2^prediction$Y_mean - 1))^2) / mean(((2^Y_val_test - 1) - mean((2^Y_val_test - 1)))^2))
            save_best_results <- best_result
          }
          
          spatial_features <- ncol(X_spatial)
          if(spatial_features > 2){
            for(colm in 3:spatial_features){
              K_spatial_tra_tra <- K_spatial_tra_tra + exp(-(pdist(X_spatial_tra[,colm, drop = FALSE], X_spatial_tra[,colm, drop = FALSE]))^2 / (2 * s_ratio_spatial^2))
              K_spatial_test_tra <- K_spatial_test_tra + exp(-(pdist(X_spatial_test[,colm, drop = FALSE], X_spatial_tra[,colm, drop = FALSE]))^2 / (2 * s_ratio_spatial^2))
              K_spatial_test_test <- K_spatial_test_test + exp(-(pdist(X_spatial_test[,colm, drop = FALSE], X_spatial_test[,colm, drop = FALSE]))^2 / (2 * s_ratio_spatial^2))
            }
          }
          
          state <- structured_gpr_train(1/(ncol(X_spatial) - 1) * K_spatial_tra_tra, 0.5 * K_temporal_tra_tra, Y_val_tra, parameters)
          prediction <- structured_gpr_test(1/(ncol(X_spatial) - 1) * K_spatial_test_tra, 1/(ncol(X_spatial) - 1) * K_spatial_test_test, 0.5 * K_temporal_test_tra, 0.5 * K_temporal_test_test, state)
          
          
          result <- sqrt(mean(((2^Y_val_test - 1) - round(2^prediction$Y_mean - 1))^2) / mean(((2^Y_val_test - 1) - mean((2^Y_val_test - 1)))^2))
          
          results_table <- c(results_table, result)
        }
        
        results_table[results_table == Inf] <- 1
        if(min(results_table) < best_result){
          best_result <- min(results_table)
          save_best_results <- c(save_best_results, best_result)
          add_feature <- all_features[which.min(results_table)]
          coords <- cbind(coords, X_spatial_all[selected_county, add_feature, drop = FALSE])
          colnames(coords)[ncol(coords)] <- all_features[which.min(results_table)]
          
          all_features <- all_features[-which.min(results_table)]
        }else
        {
          break
        }
      }
      
      cv_selected_features[[sprintf("%g", sigma)]] <- setdiff(colnames(X_spatial_all), all_features)
      nrmse_matrix['nrmse', sprintf("%g", sigma)] <- best_result
    }
    
    sigma_star <- sigma_set[max.col(t(colMeans(-nrmse_matrix)), ties.method = "last")]
    parameters$sigma <- sigma_star
    use_features_to_test <- cv_selected_features[[sprintf("%g", sigma_star)]]
    temporal_train_indices <- 1:(train_size - forecast_days)
    temporal_test_indices <- (train_size - forecast_days + 1):ncol(cases)
    
    X_temporal_tra <- X_temporal[temporal_train_indices,]
    X_temporal_tra <- scale(X_temporal_tra)
    X_temporal_test <- X_temporal[temporal_test_indices, ]
    X_temporal_test <- (X_temporal_test - matrix(attr(X_temporal_tra, "scaled:center"), nrow = nrow(X_temporal_test), ncol = ncol(X_temporal_test), byrow = TRUE)) / matrix(attr(X_temporal_tra, "scaled:scale"), nrow = nrow(X_temporal_test), ncol = ncol(X_temporal_test), byrow = TRUE)
    
    K_temporal_tra_tra <- matrix(1, nrow = nrow(X_temporal_tra), ncol = nrow(X_temporal_tra))
    K_temporal_test_tra <- matrix(1, nrow = nrow(X_temporal_test), ncol = nrow(X_temporal_tra))
    K_temporal_test_test <- matrix(1, nrow = nrow(X_temporal_test), ncol = nrow(X_temporal_test))
    
    for(feature in 1:temporal_features){
      K_temporal_tra_tra <- K_temporal_tra_tra + exp(-(pdist(X_temporal_tra[,feature, drop = FALSE], X_temporal_tra[,feature, drop = FALSE]))^2 / (2 * s_ratio_temporal^2))
      K_temporal_test_tra <- K_temporal_test_tra + exp(-(pdist(X_temporal_test[,feature, drop = FALSE], X_temporal_tra[,feature, drop = FALSE]))^2 / (2 * s_ratio_temporal^2))
      K_temporal_test_test <- K_temporal_test_test + exp(-(pdist(X_temporal_test[,feature, drop = FALSE], X_temporal_test[,feature, drop = FALSE]))^2 / (2 * s_ratio_temporal^2))
    }
    
    load('../data/spatial_data/county_coordinates.RData')
    coords <- county_coordinates[selected_county, ]
    load('../data/spatial_data/X_spaial.RData')
    
    X_spatial_all <- X_spatial[, c('n_pop_2018', 'n_people_below_poverty_level_2017', 'avg_hh_size_2010', 'n_households_2010')]
    X_spatial_all[, 'n_nonwhite'] <- X_spatial[, 'p_nonwhite'] * X_spatial[, 'n_pop_2018']
    X_spatial_all[, 'n_20_65_age'] <- rowSums(X_spatial[, c('age20', 'age25', 'age30', 'age35', 'age40', 'age45', 'age50', 'age55', 'age60')])

    load('../data/spatial_data/county_coordinates.RData')
    # selected features 
    if(length(use_features_to_test) >= 1){      
      X_spatial <- cbind(coords, X_spatial_all[selected_county, use_features_to_test])
    }else{
      X_spatial <- coords
    }
    
    X_spatial_tra <- X_spatial[spatial_train_indices, ]
    rownames(X_spatial_tra) <- rownames(X_spatial)[spatial_train_indices]
    X_spatial_test <- X_spatial[spatial_test_indices, ]
    rownames(X_spatial_test) <- rownames(X_spatial)[spatial_test_indices]
    
    X_spatial_tra <- scale(X_spatial_tra)
    X_spatial_test <- (X_spatial_test - matrix(attr(X_spatial_tra, "scaled:center"), nrow = nrow(X_spatial_test), ncol = ncol(X_spatial_test), byrow = TRUE)) / matrix(attr(X_spatial_tra, "scaled:scale"), nrow = nrow(X_spatial_test), ncol = ncol(X_spatial_test), byrow = TRUE)
    
    D_latlong_tra_tra <- pdist(X_spatial_tra[,1:2, drop = FALSE], X_spatial_tra[,1:2, drop = FALSE])
    D_latlong_test_tra <- pdist(X_spatial_test[,1:2, drop = FALSE], X_spatial_tra[,1:2, drop = FALSE])
    D_latlong_test_test <- pdist(X_spatial_test[,1:2, drop = FALSE], X_spatial_test[,1:2, drop = FALSE])
    s <- s_ratio_spatial * mean(D_latlong_tra_tra)
    K_spatial_tra_tra <- exp(-D_latlong_tra_tra^2 / (2 * s^2))
    K_spatial_test_tra <- exp(-D_latlong_test_tra^2 / (2 * s^2))
    K_spatial_test_test <- exp(-D_latlong_test_test^2 / (2 * s^2))
    
    spatial_features <- ncol(X_spatial)
    if(spatial_features > 2){
      for(feature in 3:spatial_features){
        K_spatial_tra_tra <- K_spatial_tra_tra + exp(-(pdist(X_spatial_tra[,feature, drop = FALSE], X_spatial_tra[,feature, drop = FALSE]))^2 / (2 * s_ratio_spatial^2))
        K_spatial_test_tra <- K_spatial_test_tra + exp(-(pdist(X_spatial_test[,feature, drop = FALSE], X_spatial_tra[,feature, drop = FALSE]))^2 / (2 * s_ratio_spatial^2))
        K_spatial_test_test <- K_spatial_test_test + exp(-(pdist(X_spatial_test[,feature, drop = FALSE], X_spatial_test[,feature, drop = FALSE]))^2 / (2 * s_ratio_spatial^2))
      }
    }
    
    # responce data
    Y_tra <- log2(as.matrix(cases[spatial_train_indices, temporal_train_indices]) + 1)
    Y_test <- log2(as.matrix(cases[spatial_test_indices, temporal_test_indices]) + 1)
    
    state <- structured_gpr_train(1/(ncol(X_spatial) - 1) * K_spatial_tra_tra, 0.5 * K_temporal_tra_tra, Y_tra, parameters)
    prediction <- structured_gpr_test(1/(ncol(X_spatial) - 1) * K_spatial_test_tra, 1/(ncol(X_spatial) - 1) * K_spatial_test_test, 0.5 * K_temporal_test_tra, 0.5 * K_temporal_test_test, state)
    
    observed <- round(2^Y_test - 1)
    variance <- 2^prediction$Y_variance - 1
    predicted <- round(2^prediction$Y_mean - 1)
    predicted[predicted < 0] <- 0
    
    cor <- cor(matrix(observed, nrow(Y_test) * ncol(Y_test), 1), matrix(predicted, nrow(Y_test) * ncol(Y_test), 1))
    nrmse <- sqrt(mean((observed - predicted)^2) / mean((observed  - mean(observed))^2))
    
    colnames(predicted) <- rownames(X_temporal_test)
    rownames(predicted) <- rownames(X_spatial_test)
    colnames(observed) <- rownames(X_temporal_test)
    rownames(observed) <- rownames(X_spatial_test)
    
    print(sprintf('%s, PCC: %.3f NRMSE: %.3f', location, cor, nrmse))
    
    # print(parameters$sigma / mean(apply(Y_val_tra, 1, var)))
    result <- list(observed = observed, predicted = predicted, sigma = parameters$sigma, PCC_NRMSE = c(cor, nrmse), selected_features = use_features_to_test, history = save_best_results)
    save(result, file =  sprintf('./results_%s/%s%s_%s.RData', response, unit, location, train_size))
  }
}
