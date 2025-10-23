# library(xgboost)
# library(caret)
# library(pROC)
# library(ROCR)
# library(ggplot2)
#
# cross_validation_xgb <- function(k=5, data, seed=42) {
#   set.seed(seed)
#
#   folds <- createFolds(y=data$Target_label, k=k)
#   metrics <- data.frame(Sn=numeric(k), Sp=numeric(k), ACC=numeric(k),
#                         MCC=numeric(k), F1=numeric(k), Pre=numeric(k),
#                         Recall=numeric(k), AUC=numeric(k), AUPR=numeric(k))
#
#   # 用于存储所有预测结果
#   all_predictions <- data.frame()
#
#   # 准备参数
#   params <- list(
#     objective = "binary:logistic",
#     eval_metric = "logloss",
#     max_depth = 8,
#     eta = 0.2,
#     gamma = 0,
#     subsample = 0.8,
#     colsample_bytree = 0.4,
#     min_child_weight = 1
#   )
#
#   for (i in 1:k) {
#     cat("Processing fold", i, "\n")
#
#     index_train <- folds[[i]]
#     train <- data[-index_train,]
#     test <- data[index_train,]
#
#     # 分离特征和标签
#     train_x <- data.matrix(train[, -ncol(train)])
#     train_y <- as.numeric(train$Target_label) - 1  # 转换为0-1
#
#     test_x <- data.matrix(test[, -ncol(test)])
#     test_y <- as.numeric(test$Target_label) - 1
#
#     # 创建DMatrix
#     dtrain <- xgb.DMatrix(data = train_x, label = train_y)
#     dtest <- xgb.DMatrix(data = test_x, label = test_y)
#
#     # 训练模型
#     model <- xgb.train(
#       params = params,
#       data = dtrain,
#       nrounds = 100,
#       watchlist = list(train = dtrain, test = dtest),
#       early_stopping_rounds = 10,
#       verbose = 0
#     )
#
#     # 预测
#     pro <- predict(model, dtest)
#     classification <- ifelse(pro > 0.5, 1, 0)
#
#     # 收集预测结果
#     fold_predictions <- data.frame(
#       Fold = i,
#       Prob = pro,
#       Actual = test_y,
#       Predicted = classification
#     )
#     all_predictions <- rbind(all_predictions, fold_predictions)
#
#     TP <- sum(classification == 1 & test_y == 1)
#     FP <- sum(classification == 1 & test_y == 0)
#     TN <- sum(classification == 0 & test_y == 0)
#     FN <- sum(classification == 0 & test_y == 1)
#
#     metrics$Sn[i] <- TP / (TP + FN)
#     metrics$Sp[i] <- TN / (TN + FP)
#     metrics$MCC[i] <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
#     metrics$F1[i] <- (2 * TP) / (2 * TP + FP + FN)
#     metrics$ACC[i] <- (TP + TN) / (TP + TN + FP + FN)
#     metrics$Pre[i] <- TP / (TP + FP)
#     metrics$Recall[i] <- TP / (TP + FN)
#
#     # 计算AUC
#     roc_obj <- roc(test_y, pro)
#     metrics$AUC[i] <- auc(roc_obj)
#
#     # 计算AUPR
#     pred_obj <- prediction(pro, test_y)
#     aucpr <- performance(pred_obj, 'aucpr')
#     metrics$AUPR[i] <- unlist(slot(aucpr, "y.values"))
#   }
#
#   # 绘制并保存ROC曲线 (美化版PDF)
#   roc_obj <- roc(all_predictions$Actual, all_predictions$Prob)
#   roc_auc <- round(auc(roc_obj), 2)
#   roc_data <- data.frame(
#     FPR = 1 - roc_obj$specificities,
#     TPR = roc_obj$sensitivities
#   )
#
#   roc_plot <- ggplot(roc_data, aes(x = FPR, y = TPR)) +
#     geom_line(color = "#377EB8", size = 1.2, alpha = 0.8) +
#     geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey40", size = 0.8) +
#     labs(title = "ROC Curve",
#          subtitle = bquote("AUC ="~.(roc_auc)),
#          x = "False Positive Rate (1 - Specificity)",
#          y = "True Positive Rate (Sensitivity)") +
#     theme_minimal(base_size = 12) +
#     theme(
#       plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
#       plot.subtitle = element_text(hjust = 0.5, size = 14),
#       panel.grid.major = element_line(color = "grey90", size = 0.2),
#       panel.grid.minor = element_blank(),
#       panel.border = element_rect(fill = NA, color = "grey30"),
#       aspect.ratio = 1
#     ) +
#     scale_x_continuous(expand = c(0.01, 0.01)) +
#     scale_y_continuous(expand = c(0.01, 0.01))
#
#   ggsave("XGBoost_ROC_Curve.pdf", plot = roc_plot, width = 6, height = 6, device = "pdf")
#
#   # 绘制并保存PR曲线 (美化版PDF)
#   pred_obj <- prediction(all_predictions$Prob, all_predictions$Actual)
#   perf_obj <- performance(pred_obj, "prec", "rec")
#   aucpr <- performance(pred_obj, "aucpr")
#   aupr_value <- round(unlist(slot(aucpr, "y.values")), 2)
#
#   pr_data <- data.frame(
#     Recall = unlist(perf_obj@x.values),
#     Precision = unlist(perf_obj@y.values)
#   )
#
#   pr_plot <- ggplot(pr_data, aes(x = Recall, y = Precision)) +
#     geom_line(color = "#377EB8", size = 1.2, alpha = 0.8) +
#     labs(title = "Precision-Recall Curve",
#          subtitle = bquote("AUPR ="~.(aupr_value)),
#          x = "Recall (Sensitivity)",
#          y = "Precision (Positive Predictive Value)") +
#     theme_minimal(base_size = 12) +
#     theme(
#       plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
#       plot.subtitle = element_text(hjust = 0.5, size = 14),
#       panel.grid.major = element_line(color = "grey90", size = 0.2),
#       panel.grid.minor = element_blank(),
#       panel.border = element_rect(fill = NA, color = "grey30"),
#       aspect.ratio = 1
#     ) +
#     scale_x_continuous(limits = c(0, 1), expand = c(0.01, 0.01)) +
#     scale_y_continuous(limits = c(0, 1), expand = c(0.01, 0.01))
#
#   ggsave("XGBoost_PR_Curve.pdf", plot = pr_plot, width = 6, height = 6, device = "pdf")
#
#   result <- list(
#     mean_metrics = colMeans(metrics, na.rm=TRUE),
#     all_metrics = metrics,
#     predictions = all_predictions
#   )
#
#   return(result)
# }
#
# # 数据准备
# path <- paste("./bagfeature_ndata2", ".csv", sep="")
# data <- read.csv(path, header=TRUE)
# y <- c(rep("positive", 156), rep("negative", 156))
# data <- cbind(data, y)
#
# colnames(data) <- c(1:(ncol(data)-1), "Target_label")
# data$Target_label <- as.factor(data$Target_label)
#
# # 执行交叉验证
# result_XGB <- cross_validation_xgb(k=5, data=data)
#
# # 保存结果
# print(result_XGB$mean_metrics)
# write.csv(result_XGB$all_metrics, "xgb_fold_metrics2.csv", row.names = FALSE)
# write.csv(result_XGB$predictions, "xgb_all_predictions2.csv", row.names = FALSE)


library(xgboost)
library(caret)
library(pROC)
library(ROCR)
library(ggplot2)
library(boot)
library(ggpubr)
library(tidyr)
library(dplyr)
library(reshape2)

# ====================== XGBoost交叉验证函数 ======================
cross_validation_xgb <- function(k = 5, data, seed = 42, n_bootstrap = 2000) {
  set.seed(seed)

  folds <- createFolds(y = data$Target_label, k = k)
  metrics <- data.frame(
    Sn = numeric(k), Sp = numeric(k), ACC = numeric(k),
    MCC = numeric(k), F1 = numeric(k), Pre = numeric(k),
    Recall = numeric(k), AUC = numeric(k), AUPR = numeric(k)
  )

  all_predictions <- data.frame()
  all_probs <- numeric(0)
  all_actuals <- numeric(0)

  params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 8,
    eta = 0.2,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.4,
    min_child_weight = 1
  )

  for (i in 1:k) {
    cat("Processing fold", i, "\n")

    index_train <- folds[[i]]
    train <- data[-index_train, ]
    test <- data[index_train, ]

    train_x <- data.matrix(train[, -ncol(train)])
    train_y <- as.numeric(train$Target_label) - 1

    test_x <- data.matrix(test[, -ncol(test)])
    test_y <- as.numeric(test$Target_label) - 1

    dtrain <- xgb.DMatrix(data = train_x, label = train_y)
    dtest <- xgb.DMatrix(data = test_x, label = test_y)

    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = 100,
      watchlist = list(train = dtrain, test = dtest),
      early_stopping_rounds = 10,
      verbose = 0
    )

    pro <- predict(model, dtest)
    classification <- ifelse(pro > 0.5, 1, 0)

    fold_predictions <- data.frame(
      Fold = rep(i, length(pro)),
      Prob = pro,
      Actual = test_y,
      Predicted = classification
    )
    all_predictions <- rbind(all_predictions, fold_predictions)
    all_probs <- c(all_probs, pro)
    all_actuals <- c(all_actuals, test_y)

    # 计算所有指标
    conf_matrix <- table(
      factor(classification, levels = c(0, 1)),
      factor(test_y, levels = c(0, 1))
    )
    TP <- conf_matrix[2, 2]
    FP <- conf_matrix[2, 1]
    TN <- conf_matrix[1, 1]
    FN <- conf_matrix[1, 2]

    metrics$Sn[i] <- TP / (TP + FN)  # Sensitivity (Recall)
    metrics$Sp[i] <- TN / (TN + FP)  # Specificity
    metrics$ACC[i] <- (TP + TN) / sum(conf_matrix)  # Accuracy
    metrics$MCC[i] <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # Matthews Correlation Coefficient
    metrics$F1[i] <- 2 * TP / (2 * TP + FP + FN)  # F1 Score
    metrics$Pre[i] <- TP / (TP + FP)  # Precision
    metrics$Recall[i] <- TP / (TP + FN)  # Recall (Sensitivity)

    roc_obj <- roc(test_y, pro)
    metrics$AUC[i] <- auc(roc_obj)  # AUC-ROC

    pred_obj <- prediction(pro, test_y)
    aucpr <- performance(pred_obj, "aucpr")
    metrics$AUPR[i] <- unlist(slot(aucpr, "y.values"))  # AUC-PR
  }

  # ====================== 统计显著性分析 ======================
  # Bootstrap置信区间计算函数
  bootstrap_ci <- function(metric_vector, n_bootstrap = n_bootstrap, conf_level = 0.95) {
    if (length(metric_vector) == 0 || all(is.na(metric_vector))) {
      return(c(NA, NA))
    }

    boot_func <- function(data, indices) {
      mean(data[indices], na.rm = TRUE)
    }

    boot_results <- boot(metric_vector, statistic = boot_func, R = n_bootstrap)
    ci <- boot.ci(boot_results, conf = conf_level, type = "perc")

    if (is.null(ci$percent)) {
      return(c(NA, NA))
    } else {
      return(ci$percent[4:5])
    }
  }

  # 计算每个指标的统计摘要
  metric_names <- colnames(metrics)
  stats_summary <- data.frame(
    Metric = metric_names,
    Mean = numeric(length(metric_names)),
    StdDev = numeric(length(metric_names)),
    CI_lower = numeric(length(metric_names)),
    CI_upper = numeric(length(metric_names)),
    p_value = numeric(length(metric_names)),
    Significance = character(length(metric_names)),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(metric_names)) {
    metric_vector <- metrics[[metric_names[i]]]

    # 计算均值和标准差
    mean_val <- mean(metric_vector, na.rm = TRUE)
    sd_val <- sd(metric_vector, na.rm = TRUE)

    # 计算置信区间
    ci <- bootstrap_ci(metric_vector, n_bootstrap)

    # 计算p值（假设检验：指标是否显著大于随机水平）
    if (metric_names[i] %in% c("ACC", "AUC", "AUPR")) {
      # 对于准确率、AUC和AUPR，随机水平是0.5
      t_test <- t.test(metric_vector, mu = 0.5, alternative = "greater")
      p_val <- t_test$p.value
      sig <- ifelse(p_val < 0.001, "***",
                   ifelse(p_val < 0.01, "**",
                          ifelse(p_val < 0.05, "*", "NS")))
    } else if (metric_names[i] %in% c("MCC")) {
      # 对于MCC，随机水平是0
      t_test <- t.test(metric_vector, mu = 0, alternative = "greater")
      p_val <- t_test$p.value
      sig <- ifelse(p_val < 0.001, "***",
                   ifelse(p_val < 0.01, "**",
                          ifelse(p_val < 0.05, "*", "NS")))
    } else {
      # 其他指标不进行假设检验
      p_val <- NA
      sig <- "NA"
    }

    stats_summary[i, "Mean"] <- mean_val
    stats_summary[i, "StdDev"] <- sd_val
    stats_summary[i, "CI_lower"] <- ci[1]
    stats_summary[i, "CI_upper"] <- ci[2]
    stats_summary[i, "p_value"] <- p_val
    stats_summary[i, "Significance"] <- sig
  }

  # ====================== 可视化 ======================
  # 1. ROC曲线
  roc_obj <- roc(all_actuals, all_probs)
  roc_auc <- round(auc(roc_obj), 4)
  roc_data <- data.frame(
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities
  )

  roc_plot <- ggplot(roc_data, aes(x = FPR, y = TPR)) +
    geom_line(color = "#E41A1C", linewidth = 1.2, alpha = 0.8) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey40", linewidth = 0.8) +
    labs(
      title = "ROC Curve",
      subtitle = bquote("AUC = " ~ .(roc_auc) ~ " (" * 95 * "% CI: " *
                          .(round(stats_summary$CI_lower[stats_summary$Metric == "AUC"], 4)) * "-" *
                          .(round(stats_summary$CI_upper[stats_summary$Metric == "AUC"], 4)) * ")"),
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
      plot.subtitle = element_text(hjust = 0.5, size = 14, margin = margin(b = 15)),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "grey30", linewidth = 0.8),
      panel.background = element_rect(fill = "white", color = NA),
      aspect.ratio = 1,
      legend.position = "none"
    ) +
    scale_x_continuous(expand = c(0.01, 0.01), limits = c(0, 1)) +
    scale_y_continuous(expand = c(0.01, 0.01), limits = c(0, 1))

  ggsave("Sta_XGBoost_ROC_Curve.pdf", plot = roc_plot, width = 7, height = 7, device = "pdf")

  # 2. PR曲线
  pred_obj <- prediction(all_probs, all_actuals)
  perf_obj <- performance(pred_obj, "prec", "rec")
  aucpr <- performance(pred_obj, "aucpr")
  aupr_value <- round(unlist(slot(aucpr, "y.values")), 4)

  pr_data <- data.frame(
    Recall = unlist(perf_obj@x.values),
    Precision = unlist(perf_obj@y.values)
  )

  pr_plot <- ggplot(pr_data, aes(x = Recall, y = Precision)) +
    geom_line(color = "#4DAF4A", linewidth = 1.2, alpha = 0.8) +
    labs(
      title = "Precision-Recall Curve",
      subtitle = bquote("AUPR = " ~ .(aupr_value) ~ " (" * 95 * "% CI: " *
                          .(round(stats_summary$CI_lower[stats_summary$Metric == "AUPR"], 4)) * "-" *
                          .(round(stats_summary$CI_upper[stats_summary$Metric == "AUPR"], 4)) * ")"),
      x = "Recall (Sensitivity)",
      y = "Precision (Positive Predictive Value)"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
      plot.subtitle = element_text(hjust = 0.5, size = 14, margin = margin(b = 15)),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "grey30", linewidth = 0.8),
      panel.background = element_rect(fill = "white", color = NA),
      aspect.ratio = 1
    ) +
    scale_x_continuous(limits = c(0, 1), expand = c(0.01, 0.01)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0.01, 0.01))

  ggsave("Sta_XGBoost_PR_Curve.pdf", plot = pr_plot, width = 7, height = 7, device = "pdf")

  # 3. 性能指标可视化（所有指标）
  metrics_long <- reshape2::melt(metrics)
  colnames(metrics_long) <- c("Metric", "Value")

  metrics_plot <- ggplot(metrics_long, aes(x = Metric, y = Value, fill = Metric)) +
    geom_boxplot(alpha = 0.8, outlier.size = 2) +
    geom_jitter(width = 0.1, size = 2, alpha = 0.6) +
    stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "red") +
    labs(title = "Model Performance Metrics Across Folds",
         subtitle = "Red diamonds indicate mean values",
         x = "Performance Metric",
         y = "Value") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none",
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "grey30")
    ) +
    scale_fill_brewer(palette = "Set3")

  ggsave("XGBoost_Performance_Metrics.pdf", plot = metrics_plot, width = 10, height = 7, device = "pdf")

  # 4. 统计显著性表格可视化
  stats_table <- ggtexttable(stats_summary[, 1:7], rows = NULL,
                            theme = ttheme(
                              base_style = "light",
                              base_size = 10,
                              padding = unit(c(4, 4), "mm"),
                              colnames.style = colnames_style(color = "white", fill = "#377EB8"),
                              tbody.style = tbody_style(color = "black", fill = c("#F0F0F0", "#FFFFFF"))
                            )) +
    labs(title = "Statistical Summary of Performance Metrics") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))

  ggsave("XGBoost_Statistical_Summary.pdf", plot = stats_table, width = 12, height = 8, device = "pdf")

  # ====================== 结果返回 ======================
  result <- list(
    mean_metrics = colMeans(metrics, na.rm = TRUE),
    all_metrics = metrics,
    predictions = all_predictions,
    stats_summary = stats_summary,
    roc_auc = roc_auc,
    aupr_value = aupr_value
  )

  return(result)
}

# ====================== 数据准备和执行 ======================
# 数据准备
path <- paste0("./bagfeature_ndata2", ".csv")
data <- read.csv(path, header = TRUE)

# 确保数据平衡
positive_samples <- 156
negative_samples <- 156
y <- c(rep("positive", positive_samples), rep("negative", negative_samples))
data <- cbind(data, y)

colnames(data) <- c(1:(ncol(data) - 1), "Target_label")
data$Target_label <- as.factor(data$Target_label)

# 执行交叉验证
result_XGB <- cross_validation_xgb(k = 5, data = data)

# 保存结果
print(result_XGB$mean_metrics)
print(result_XGB$stats_summary)

write.csv(result_XGB$all_metrics, "xgb_fold_metrics.csv", row.names = FALSE)
write.csv(result_XGB$predictions, "xgb_all_predictions.csv", row.names = FALSE)
write.csv(result_XGB$stats_summary, "xgb_statistical_summary.csv", row.names = FALSE)

# ====================== 模型比较 ======================
# 假设已有另一个模型的结果 (例如SVM模型)
# 这里用模拟数据演示，实际使用时替换为真实结果
set.seed(42)
result_other <- data.frame(
#   Sn = rnorm(5, mean = 0.82, sd = 0.04),
  Sp = rnorm(5, mean = 0.85, sd = 0.03),
  ACC = rnorm(5, mean = 0.83, sd = 0.03),
  MCC = rnorm(5, mean = 0.67, sd = 0.05),
#   F1 = rnorm(5, mean = 0.83, sd = 0.04),
  Pre = rnorm(5, mean = 0.82, sd = 0.04),
#   Recall = rnorm(5, mean = 0.82, sd = 0.04),
  AUC = rnorm(5, mean = 0.90, sd = 0.02),
  AUPR = rnorm(5, mean = 0.91, sd = 0.03)
)

# ====================== 模型比较：统计显著性检验 ======================
# 定义要比较的指标（所有指标）
common_metrics <- c("Sp", "ACC", "MCC", "Pre", "AUC", "AUPR")

model_comparison <- data.frame(
  Metric = character(),
  t_statistic = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

# 提取XGBoost模型的指标
xgb_metrics <- result_XGB$all_metrics

# 对每个指标进行配对t检验
for (metric in common_metrics) {
  # 确保两个模型都有该指标
  if (!(metric %in% colnames(xgb_metrics)) || !(metric %in% colnames(result_other))) {
    next
  }

  # 确保两个模型有相同数量的折叠
  if (nrow(xgb_metrics) != nrow(result_other)) {
    stop("两个模型的折叠数不一致，无法进行配对检验")
  }

  t_test <- t.test(
    xgb_metrics[[metric]],
    result_other[[metric]],
    paired = TRUE
  )

  model_comparison <- rbind(model_comparison, data.frame(
    Metric = metric,
    t_statistic = t_test$statistic,
    p_value = t_test$p.value
  ))
}

# 添加显著性标记
model_comparison$Significance <- ifelse(model_comparison$p_value < 0.001, "***",
                                       ifelse(model_comparison$p_value < 0.01, "**",
                                              ifelse(model_comparison$p_value < 0.05, "*", "NS")))

print(model_comparison)
write.csv(model_comparison, "XGB_vs_Other_Model_Comparison.csv", row.names = FALSE)

# ====================== 模型比较：可视化 ======================
# 准备数据
xgb_data <- xgb_metrics
xgb_data$Model <- "Our"
xgb_data$Fold <- 1:5

other_data <- result_other
other_data$Model <- "TransRM"
other_data$Fold <- 1:5

combined_data <- rbind(
  xgb_data %>% select(Model, Fold, all_of(common_metrics)),
  other_data %>% select(Model, Fold, all_of(common_metrics))
)

# 转换为长格式
long_data <- combined_data %>%
  pivot_longer(
    cols = all_of(common_metrics),
    names_to = "Metric",
    values_to = "Value"
  )

# 为每个指标计算位置信息
annotation_data <- model_comparison %>%
  mutate(
    y_pos = max(long_data$Value, na.rm = TRUE) * 0.95,
    x_min = 0.8,
    x_max = 2.2
  )

# 创建自定义颜色方案
model_colors <- c("Our" = "#E41A1C", "TransRM" = "#377EB8")

# 绘制比较箱线图（所有指标）
comparison_plot <- ggplot(long_data, aes(x = Model, y = Value, fill = Model)) +
  geom_boxplot(width = 0.6, alpha = 0.8, outlier.size = 2) +
  geom_point(aes(group = Model), position = position_jitterdodge(jitter.width = 0.2),
             size = 2, alpha = 0.6) +
  facet_wrap(~ Metric, scales = "free_y", nrow = 3, ncol = 3,
             labeller = labeller(Metric = c(
#                Sn = "Sensitivity",
               Sp = "Specificity",
               ACC = "Accuracy",
               MCC = "MCC",
#                F1 = "F1 Score",
               Pre = "Precision",
#                Recall = "Recall",
               AUC = "AUC-ROC",
               AUPR = "AUPR"
             ))) +
  geom_signif(
    data = annotation_data,
    aes(xmin = x_min, xmax = x_max, annotations = Significance, y_position = y_pos),
    manual = TRUE, textsize = 6, vjust = -0.2,
    inherit.aes = FALSE
  ) +
  scale_fill_manual(values = model_colors) +
  labs(title = "Model Performance Comparison",
#        subtitle = "XGBoost vs. Other Model (5-fold cross-validation results)",
       y = "Metric Value") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    plot.subtitle = element_text(hjust = 0.5, size = 14, margin = margin(b = 15)),
    strip.text = element_text(face = "bold", size = 12),
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank(),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA, color = "grey30", linewidth = 0.5),
    strip.background = element_rect(fill = "grey95", color = "grey30", linewidth = 0.5)
  ) +
  guides(fill = guide_legend(nrow = 1))

# 保存图表
ggsave("XGB_vs_Other_Model_Comparison.pdf", plot = comparison_plot,
       width = 12, height = 10, device = "pdf")

# 显示图表
print(comparison_plot)

# ====================== 输出最终结果 ======================
cat("\n\n====== XGBoost 模型评估完成 ======\n")
cat("平均性能指标:\n")
print(result_XGB$mean_metrics)

cat("\n\n====== 模型比较结果 ======\n")
cat("XGBoost vs. 其他模型:\n")
print(model_comparison)





# library(xgboost)
# library(caret)
# library(pROC)
# library(ROCR)
# library(ggplot2)
# library(randomForest)
# library(lightgbm)
# library(e1071)
# library(glmnet)
#
# # 统一交叉验证函数
# cross_validation_model <- function(model_name, k=5, data, seed=42) {
#   set.seed(seed)
#   folds <- createFolds(y=data$Target_label, k=k)
#
#   metrics <- data.frame(Sn=numeric(k), Sp=numeric(k), ACC=numeric(k),
#                         MCC=numeric(k), F1=numeric(k), Pre=numeric(k),
#                         Recall=numeric(k), AUC=numeric(k), AUPR=numeric(k))
#
#   all_predictions <- data.frame()
#
#   for (i in 1:k) {
#     cat("Processing", model_name, "- fold", i, "\n")
#
#     index_train <- folds[[i]]
#     train <- data[-index_train, ]
#     test <- data[index_train, ]
#
#     train_x <- data.matrix(train[, -ncol(train)])
#     train_y <- as.numeric(train$Target_label) - 1
#
#     test_x <- data.matrix(test[, -ncol(test)])
#     test_y <- as.numeric(test$Target_label) - 1
#
#     # 模型训练和预测
#     pro <- switch(model_name,
#       "XGBoost" = {
#         dtrain <- xgb.DMatrix(data = train_x, label = train_y)
#         dtest <- xgb.DMatrix(data = test_x, label = test_y)
#
#         params <- list(
#           objective = "binary:logistic",
#           eval_metric = "logloss",
#           max_depth = 8,
#           eta = 0.2,
#           gamma = 0,
#           subsample = 0.8,
#           colsample_bytree = 0.4,
#           min_child_weight = 1
#         )
#
#         watchlist <- list(train = dtrain, eval = dtest)
#
#         model <- xgb.train(
#           params = params,
#           data = dtrain,
#           nrounds = 100,
#           watchlist = watchlist,
#           early_stopping_rounds = 10,
#           verbose = 0
#         )
#         predict(model, dtest)
#       },
#       "RF" = {
#         train_df <- data.frame(train_x)
#         train_df$Target_label <- factor(train_y)
#
#         model <- randomForest(
#           Target_label ~ .,
#           data = train_df,
#           ntree = 10,
#           mtry = floor(sqrt(ncol(train_x))),
#           importance = TRUE,
#           replace = TRUE,
#           sampsize = floor(0.8 * nrow(train_x)),
#           nodesize = 1000,
#           maxnodes = NULL,
#           do.trace = FALSE,
#           strata = train_df$Target_label,
#           keep.forest = TRUE
#         )
#         predict(model, data.frame(test_x), type = "prob")[, 2]
#       },
#       "LightGBM" = {
#         dtrain <- lgb.Dataset(data = train_x, label = train_y)
#         dtest <- lgb.Dataset.create.valid(dtrain, data = test_x, label = test_y)
#
#         params <- list(
#           objective = "binary",
#           metric = "binary_logloss",
#           num_leaves = 10,
#           max_depth = 12,
#           learning_rate = 0.8,
#           feature_fraction = 0.2,
#           min_data_in_leaf = 2,
#           min_gain_to_split = 0.01,
#           max_bin = 255,
#           scale_pos_weight = 1,
#           num_threads = 4
#         )
#
#         model <- lgb.train(
#           params,
#           dtrain,
#           100,
#           valids = list(test = dtest),
#           early_stopping_rounds = 10,
#           verbose = -1
#         )
#         predict(model, test_x)
#       },
#       "LR" = {
#         cv_model <- cv.glmnet(
#           train_x,
#           train_y,
#           family = "binomial",
#           alpha = 1,
#           type.measure = "auc",
#           parallel = TRUE,
#           standardize = TRUE,
#           intercept = TRUE,
#           maxit = 50,
#           thresh = 1e-07
#         )
#         predict(cv_model, newx = test_x, s = "lambda.min", type = "response")[, 1]
#       },
#       "SVM" = {
#         train_df <- data.frame(train_x)
#         train_df$Target_label <- factor(train_y)
#
#         model <- svm(
#           Target_label ~ .,
#           data = train_df,
#           kernel = "radial",
#           cost = 0.9,
#           gamma = 0.001,
#           coef0 = 0,
#           degree = 10,
#           probability = TRUE
#         )
#
#         pred_svm <- predict(model, data.frame(test_x), probability = TRUE)
#         attr(pred_svm, "probabilities")[, "1"]
#       }
#     )
#
#     classification <- ifelse(pro > 0.5, 1, 0)
#
#     # 收集预测结果
#     fold_predictions <- data.frame(
#       Fold = i,
#       Prob = pro,
#       Actual = test_y,
#       Predicted = classification,
#       Model = model_name
#     )
#     all_predictions <- rbind(all_predictions, fold_predictions)
#
#     # 计算指标
#     TP <- sum(classification == 1 & test_y == 1)
#     FP <- sum(classification == 1 & test_y == 0)
#     TN <- sum(classification == 0 & test_y == 0)
#     FN <- sum(classification == 0 & test_y == 1)
#
#     metrics$Sn[i] <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
#     metrics$Sp[i] <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
#
#     denom <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
#     metrics$MCC[i] <- ifelse(denom > 0, (TP * TN - FP * FN) / denom, 0)
#
#     metrics$F1[i] <- ifelse((2 * TP + FP + FN) > 0, (2 * TP) / (2 * TP + FP + FN), 0)
#     metrics$ACC[i] <- (TP + TN) / (TP + TN + FP + FN)
#     metrics$Pre[i] <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
#     metrics$Recall[i] <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
#
#     # 计算AUC
#     if (length(unique(test_y)) > 1) {
#       roc_obj <- roc(test_y, pro)
#       metrics$AUC[i] <- auc(roc_obj)
#     } else {
#       metrics$AUC[i] <- NA
#     }
#
#     # 计算AUPR
#     pred_obj <- prediction(pro, test_y)
#     aucpr <- performance(pred_obj, 'aucpr')
#     metrics$AUPR[i] <- ifelse(length(slot(aucpr, "y.values")) > 0, unlist(slot(aucpr, "y.values")), NA)
#   }
#
#   list(
#     mean_metrics = colMeans(metrics, na.rm = TRUE),
#     all_metrics = metrics,
#     predictions = all_predictions
#   )
# }
#
# # 数据准备
# path <- paste("./bagfeature_ndata2", ".csv", sep = "")
# data <- read.csv(path, header = TRUE)
# y <- c(rep("positive", 156), rep("negative", 156))
# data <- cbind(data, y)
#
# colnames(data) <- c(1:(ncol(data) - 1), "Target_label")
# data$Target_label <- as.factor(data$Target_label)
#
# # 定义要比较的模型
# models <- c("XGBoost", "RF", "LightGBM", "LR", "SVM")
# all_results <- list()
#
# # 执行交叉验证
# for (model in models) {
#   result <- cross_validation_model(model, k = 5, data = data)
#   all_results[[model]] <- result
#
#   # 保存结果
#   write.csv(result$all_metrics, paste0(tolower(model), "_fold_metrics.csv"), row.names = FALSE)
#   write.csv(result$predictions, paste0(tolower(model), "_all_predictions.csv"), row.names = FALSE)
#   cat("\n", model, "Mean Metrics:\n")
#   print(result$mean_metrics)
# }
#
# # 合并所有模型的预测结果
# combined_preds <- do.call(rbind, lapply(all_results, function(x) x$predictions))
#
# # 计算每个模型的平均AUC和AUPR
# model_metrics <- data.frame(
#   Model = models,
#   AUC = sapply(models, function(m) mean(all_results[[m]]$all_metrics$AUC, na.rm = TRUE)),
#   AUPR = sapply(models, function(m) mean(all_results[[m]]$all_metrics$AUPR, na.rm = TRUE))
# )
#
# # 创建ROC曲线数据（指标包含在模型名称中）
# roc_plots <- lapply(models, function(m) {
#   df <- combined_preds[combined_preds$Model == m, ]
#   if (length(unique(df$Actual)) > 1) {
#     roc_obj <- roc(df$Actual, df$Prob)
#     auc_value <- round(auc(roc_obj), 2)
#     data.frame(
#       FPR = 1 - roc_obj$specificities,
#       TPR = roc_obj$sensitivities,
#       Model = paste0(m, " (AUC=", auc_value, ")")
#     )
#   } else {
#     NULL
#   }
# })
#
# roc_data <- do.call(rbind, roc_plots)
#
# # 创建PR曲线数据（指标包含在模型名称中）
# pr_plots <- lapply(models, function(m) {
#   df <- combined_preds[combined_preds$Model == m, ]
#   if (length(unique(df$Actual)) > 1) {
#     pred_obj <- prediction(df$Prob, df$Actual)
#     perf_obj <- performance(pred_obj, "prec", "rec")
#     aucpr <- performance(pred_obj, "aucpr")
#     aupr_value <- round(unlist(slot(aucpr, "y.values")), 2)
#     data.frame(
#       Recall = unlist(perf_obj@x.values),
#       Precision = unlist(perf_obj@y.values),
#       Model = paste0(m, " (AUPR=", aupr_value, ")")
#     )
#   } else {
#     NULL
#   }
# })
#
# pr_data <- do.call(rbind, pr_plots)
#
# # 绘制ROC曲线
# roc_plot <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
#   geom_line(size = 1.2, alpha = 0.8) +
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey40", size = 0.8) +
#   scale_color_brewer(palette = "Set1") +
#   labs(title = "ROC Curves Comparison",
#        x = "False Positive Rate (1 - Specificity)",
#        y = "True Positive Rate (Sensitivity)") +
#   theme_minimal(base_size = 12) +
#   theme(
#     plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
#     legend.position = "bottom",
#     legend.title = element_blank(),
#     panel.grid.major = element_line(color = "grey90", size = 0.2),
#     panel.grid.minor = element_blank(),
#     panel.border = element_rect(fill = NA, color = "grey30", size = 0.5),
#     aspect.ratio = 1
#   ) +
#   guides(color = guide_legend(nrow = 2, byrow = TRUE))
#
# ggsave("combined_roc_curves.pdf", plot = roc_plot, width = 8, height = 9, device = "pdf")
#
# # 绘制PR曲线
# pr_plot <- ggplot(pr_data, aes(x = Recall, y = Precision, color = Model)) +
#   geom_line(size = 1.2, alpha = 0.8) +
#   scale_color_brewer(palette = "Set1") +
#   labs(title = "Precision-Recall Curves Comparison",
#        x = "Recall (Sensitivity)",
#        y = "Precision (Positive Predictive Value)") +
#   theme_minimal(base_size = 12) +
#   theme(
#     plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
#     legend.position = "bottom",
#     legend.title = element_blank(),
#     panel.grid.major = element_line(color = "grey90", size = 0.2),
#     panel.grid.minor = element_blank(),
#     panel.border = element_rect(fill = NA, color = "grey30", size = 0.5),
#     aspect.ratio = 1
#   ) +
#   guides(color = guide_legend(nrow = 2, byrow = TRUE))
#
# ggsave("combined_pr_curves.pdf", plot = pr_plot, width = 8, height = 9, device = "pdf")
#
# # 输出汇总指标
# cat("\nModel Comparison Summary:\n")
# print(model_metrics)
# write.csv(model_metrics, "model_comparison_summary.csv", row.names = FALSE)
#
# # 创建性能指标表格
# performance_table <- data.frame(
#   Model = models,
#   Sn = sapply(models, function(m) mean(all_results[[m]]$all_metrics$Sn, na.rm = TRUE)),
#   Sp = sapply(models, function(m) mean(all_results[[m]]$all_metrics$Sp, na.rm = TRUE)),
#   ACC = sapply(models, function(m) mean(all_results[[m]]$all_metrics$ACC, na.rm = TRUE)),
#   MCC = sapply(models, function(m) mean(all_results[[m]]$all_metrics$MCC, na.rm = TRUE)),
#   F1 = sapply(models, function(m) mean(all_results[[m]]$all_metrics$F1, na.rm = TRUE)),
#   AUC = sapply(models, function(m) mean(all_results[[m]]$all_metrics$AUC, na.rm = TRUE)),
#   AUPR = sapply(models, function(m) mean(all_results[[m]]$all_metrics$AUPR, na.rm = TRUE))
# )
#
# # 打印性能指标表格
# cat("\nPerformance Metrics Summary:\n")
# print(performance_table)
# write.csv(performance_table, "performance_metrics_summary.csv", row.names = FALSE)


# library(randomForest)
# library("caret")
# library(pROC)
# #k: k fold
# #data: the data used for cross validataion
# #dir: the directory where the results are saved
# #filename: the result files name
# cross_validation_rf <- function(k=5,data){
#
#   #train
#   n=length(names(data))
#   folds<-createFolds(y=data$Target_label,k=k) #����training��laber-Species�����ݼ��зֳ�k�ȷ�
#   TPR=c(length=k)
#   FPR=c(length=k)
#   MCC=c(length=k)
#   F1=c(length=k)
#   ACC=c(length=k)
#   precision=c(length=k)
#   recall=c(length=k)
#   ROCArea=c(length=k)
#   AUPR=c(length=k)
#   #PRCArea=c(length=k)
#   prob=c(length=312)
#   for(i in 1:k){
#     print(paste(i,"-fold"))
#     index=1:nrow(data)
#     index_train=sample(index[-folds[[i]]])
#     train<-data[index_train,]
#     index_test=sample(index[folds[[i]]])
#     test<-data[index_test,]
#     #    y_test<-y_label[index_test]
#     rf <- randomForest(Target_label ~ ., data=train)
#     classification=predict(rf,newdata=test)
#     pro=predict(rf,newdata=test,type="prob")
#     prob[index_test]=predict(rf,newdata=test,type="prob")
#     TP <- as.numeric(sum(classification=="positive" & test$Target_label=="positive"))
#     FP <- as.numeric(sum(classification=="positive" & test$Target_label=="negative"))
#     TN <- as.numeric(sum(classification=="negative" & test$Target_label=="negative"))
#     FN <- as.numeric(sum(classification=="negative" & test$Target_label=="positive"))
#     TPR[i]=TP/(TP+FN)
#     FPR[i]=TN/(TN+FP)
#     MCC[i]=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
#     F1[i]=(2*TP)/(2*TP+FP+FN)
#     ACC[i]=(TP+TN)/(TP+TN+FP+FN)
#     precision[i]=TP/(TP+FP)
#     recall[i]=TP/(TP+FN)
#     ROCArea[i]=auc(roc(test$Target_label,pro[,2]))
#     pred <-prediction(pro[,2],test$Target_label)
#     perf <- performance(pred,"prec","rec")
#     plot(perf, col='blue',lty=2)
#     aucpr <- performance(pred,'aucpr')
#     AUPR[i] = unlist(slot(aucpr,"y.values"))
#     #    AUPR[i]=AUC(obs=y_test,pred=pro[,2],curve = "PR", simplif=TRUE, main = "PR curve")
#   }
#   result=data.frame(mean(TPR),mean(FPR),mean(ACC),mean(MCC),mean(F1),mean(precision),mean(recall),mean(ROCArea),mean(AUPR))
#   colnames(result)<-c("Sn","Sp","ACC","MCC","F1","Pre","Recall","AUC","AUPR")
#   # file1=paste(dir,paste(filename,".result.csv"),sep = '/')
#   # write.csv(result,paste(dir,paste(filename,k,".result.rf.csv"),sep = '/'),row.names = F)
#   #  write.csv(prob,"E:/m6A-circRNA/data/test_data/data_m6a/prob_rf.csv",row.names = F)
#
#   return(result)
# }
# library(randomForest)
# library("caret")
# library(pROC)
# library(ROCR)
#
# path = paste("./bagfeature_ndata_origin",".csv",sep="")
# data = read.csv(path,header=T)
# y=c(rep("positive",156),rep("negative",156))
# data=cbind(data,y)
#
# colnames(data)=c(1:(ncol(data)-1),"Target_label")
#
# data$Target_label <- as.factor(data$Target_label)
#
# write.csv(data,"./features_all_origin.csv",row.names = F)
# data = read.csv("./features_all_origin.csv",header=T)
#
# data$Target_label <- as.factor(data$Target_label)
# #data=read.csv("E:/m6A-circRNA/data/test_data/data_m6a/features_mid.csv",header = T)
# # result_R=as.data.frame(array(,dim=c(10,9)))
# # result_S=as.data.frame(array(,dim=c(10,9)))
# # result_K=as.data.frame(array(,dim=c(10,9)))
# # result_L=as.data.frame(array(,dim=c(10,9)))
# # result_X=as.data.frame(array(,dim=c(10,9)))
# result_R=cross_validation_rf(k=5,data=data)
# print(result_R)
#
# library(randomForest)
# library(caret)
# library(pROC)
# library(ROCR)
#
# cross_validation_rf <- function(k=5, data){
#   n <- nrow(data)
#   folds <- createFolds(y = data$Target_label, k = k)
#
#   # 初始化结果存储
#   fold_results <- list()
#   TPR <- numeric(k)
#   FPR <- numeric(k)  # 修正：存储FPR而非Sp
#   MCC <- numeric(k)
#   F1 <- numeric(k)
#   ACC <- numeric(k)
#   precision <- numeric(k)
#   recall <- numeric(k)
#   ROCArea <- numeric(k)
#   AUPR <- numeric(k)
#
#   # 存储所有样本的概率和真实标签
#   all_probs <- numeric(n)
#   all_labels <- character(n)
#
#   for(i in 1:k){
#     cat("\n=========== Fold", i, "===========\n")
#
#     # 创建训练集和测试集
#     test_index <- folds[[i]]
#     train_index <- setdiff(1:n, test_index)
#
#     train <- data[train_index, ]
#     test <- data[test_index, ]
#
#     # 训练模型
#     rf <- randomForest(Target_label ~ ., data = train)
#
#     # 预测
#     classification <- predict(rf, newdata = test)
#     pro <- predict(rf, newdata = test, type = "prob")
#
#     # 存储概率和标签
#     all_probs[test_index] <- pro[, "positive"]
#     all_labels[test_index] <- as.character(test$Target_label)
#
#     # 计算混淆矩阵
#     conf_matrix <- table(Predicted = classification, Actual = test$Target_label)
#     print(conf_matrix)
#
#     TP <- conf_matrix["positive", "positive"]
#     FP <- conf_matrix["positive", "negative"]
#     TN <- conf_matrix["negative", "negative"]
#     FN <- conf_matrix["negative", "positive"]
#
#     # 计算指标
#     TPR[i] <- TP / (TP + FN)  # Sensitivity/Recall
#     FPR[i] <- FP / (FP + TN)  # 修正：真正的FPR
#     precision[i] <- TP / (TP + FP)
#     recall[i] <- TPR[i]       # Recall = TPR
#     ACC[i] <- (TP + TN) / (TP + TN + FP + FN)
#     F1[i] <- 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
#
#     # 计算MCC（添加分母保护）
#     denom <- sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
#     MCC[i] <- ifelse(denom > 0, (TP*TN - FP*FN)/denom, 0)
#
#     # 计算AUC
#     ROCArea[i] <- auc(roc(test$Target_label, pro[, "positive"]))
#
#     # 计算AUPR
#     pred_obj <- prediction(pro[, "positive"], test$Target_label,
#                            label.ordering = c("negative", "positive"))
#     aucpr <- performance(pred_obj, "aucpr")
#     AUPR[i] <- unlist(slot(aucpr, "y.values"))
#
#     # 打印当前折结果
#     cat("Fold", i, "Metrics:\n")
#     cat(sprintf(" Sn/TPR: %.4f", TPR[i]), "\n")
#     cat(sprintf(" FPR: %.4f", FPR[i]), "\n")
#     cat(sprintf(" Sp: %.4f", TN/(TN+FP)), "\n")  # 特异性
#     cat(sprintf(" ACC: %.4f", ACC[i]), "\n")
#     cat(sprintf(" MCC: %.4f", MCC[i]), "\n")
#     cat(sprintf(" F1: %.4f", F1[i]), "\n")
#     cat(sprintf(" Precision: %.4f", precision[i]), "\n")
#     cat(sprintf(" Recall: %.4f", recall[i]), "\n")
#     cat(sprintf(" AUC: %.4f", ROCArea[i]), "\n")
#     cat(sprintf(" AUPR: %.4f", AUPR[i]), "\n\n")
#
#     # 存储当前折结果
#     fold_results[[i]] <- data.frame(
#       Fold = i,
#       Sn = TPR[i],
#       FPR = FPR[i],
#       Sp = TN/(TN+FP),
#       ACC = ACC[i],
#       MCC = MCC[i],
#       F1 = F1[i],
#       Precision = precision[i],
#       Recall = recall[i],
#       AUC = ROCArea[i],
#       AUPR = AUPR[i]
#     )
#   }
#
#   # 计算平均结果
#   mean_result <- data.frame(
#     Sn = mean(TPR),
#     FPR = mean(FPR),
#     Sp = mean(sapply(fold_results, function(x) x$Sp)),
#     ACC = mean(ACC),
#     MCC = mean(MCC),
#     F1 = mean(F1),
#     Precision = mean(precision),
#     Recall = mean(recall),
#     AUC = mean(ROCArea),
#     AUPR = mean(AUPR)
#   )
#
#   # 整体性能评估
#   cat("=====================\n")
#   cat("Overall Performance:\n")
#   overall_auc <- auc(roc(all_labels, all_probs))
#   pred_obj <- prediction(all_probs, all_labels, label.ordering = c("negative", "positive"))
#   overall_aupr <- unlist(performance(pred_obj, "aucpr")@y.values)
#
#   cat(sprintf(" Overall AUC: %.4f\n", overall_auc))
#   cat(sprintf(" Overall AUPR: %.4f\n", overall_aupr))
#
#   # 返回结果
#   return(list(
#     fold_results = do.call(rbind, fold_results),
#     mean_result = mean_result,
#     overall_metrics = c(AUC = overall_auc, AUPR = overall_aupr)
#   ))
# }
#
# # 数据准备
# path <- "./bagfeature_ndata_origin.csv"
# data <- read.csv(path, header = TRUE)
# y <- factor(c(rep("positive", 156), rep("negative", 156)),
#             levels = c("positive", "negative"))
# data <- cbind(data, Target_label = y)
#
# # 运行交叉验证
# set.seed(123)  # 确保结果可重现
# results <- cross_validation_rf(k = 5, data = data)
#
# # 输出详细结果
# cat("\n===== Detailed Fold Results =====\n")
# print(results$fold_results)
#
# cat("\n===== Average Metrics =====\n")
# print(results$mean_result)
#
# cat("\n===== Overall AUC/AUPR =====\n")
# print(results$overall_metrics)