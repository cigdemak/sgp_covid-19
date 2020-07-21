structured_gpr_train <- function(K_1_tra_tra, K_2_tra_tra, Y_tra, parameters) {
  
  svd_1 <- eigen(K_1_tra_tra)
  svd_2 <- eigen(K_2_tra_tra)
  state <- list(U_1 = svd_1$vectors, d_1 = svd_1$values, U_2 = svd_2$vectors, d_2 = svd_2$values, Y_tra = Y_tra, parameters = parameters)
}
