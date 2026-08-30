# Q2 — MLflow Experiment Comparison: Analysis

## Screenshot

`results/mlflow_comparison.png` — MLflow comparison table for all 6 runs in the `mnist-mlp` experiment.

## Written analysis (150–250 words)

The best-performing run was **`mlp-h128_64-lr0.001-bs128`**. It achieved a **test accuracy of 0.9731**, **test macro-F1 of 0.9730**, and **validation accuracy of 0.9724**. This run performed slightly better than the larger `(256, 128, 64)` architecture at the same learning rate, suggesting that the additional network capacity did not provide a meaningful improvement for this MNIST configuration.

The MLflow training curves provide evidence of some overfitting behaviour. Across the runs, `train_loss` generally decreases as training progresses, while `val_accuracy_epoch` tends to plateau or fluctuate rather than continuously improving. This gap is more noticeable in the higher-learning-rate (`0.01`) runs, where validation performance is less stable while training loss continues to fall. Therefore, the model can continue fitting the training data without obtaining corresponding validation improvements.

The **learning rate had the larger effect on performance**. Holding the architecture fixed, changing the learning rate from `0.001` to `0.01` reduced test accuracy by approximately 0.0070 for `(64,)`, 0.0179 for `(128, 64)`, and 0.0081 for `(256, 128, 64)`. In comparison, changing architecture at a fixed learning rate produced smaller accuracy changes. The results indicate that `0.01` was too aggressive for these MLP configurations, while `0.001` produced more consistent generalization.

## Code added to the starter script (Q2.3)

### Hyperparameters logged

```python
mlflow.log_param("model_type", "MLPClassifier")
mlflow.log_param("dataset", "MNIST")
mlflow.log_param("hidden_layer_sizes", str(hidden_layer_sizes))
mlflow.log_param("learning_rate_init", learning_rate)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("alpha", 0.0001)
mlflow.log_param("activation", "relu")
mlflow.log_param("solver", "adam")
mlflow.log_param("max_iter", max_iter)
mlflow.log_param("random_state", 42)
```

### Per-epoch metrics logged

```python
for epoch, loss in enumerate(model.loss_curve_):
    mlflow.log_metric("train_loss", loss, step=epoch)

if hasattr(model, "validation_scores_"):
    for epoch, val_acc_epoch in enumerate(model.validation_scores_):
        mlflow.log_metric(
            "val_accuracy_epoch",
            val_acc_epoch,
            step=epoch
        )
```

### Final metrics logged

```python
mlflow.log_metric("val_accuracy", val_accuracy)
mlflow.log_metric("test_accuracy", test_accuracy)
mlflow.log_metric("test_f1_macro", test_f1)
mlflow.log_metric("n_epochs_run", len(model.loss_curve_))
mlflow.log_metric("train_time_sec", train_time)
```

## Experiment results

| Run | Learning rate | Hidden layers | Batch size | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---|---:|---:|---:|---:|
| mlp-h256_128_64-lr0.01-bs128 | 0.01 | (256, 128, 64) | 128 | 0.9639 | 0.9642 | 0.9641 |
| mlp-h128_64-lr0.01-bs128 | 0.01 | (128, 64) | 128 | 0.9555 | 0.9552 | 0.9556 |
| mlp-h64-lr0.01-bs128 | 0.01 | (64,) | 128 | 0.9628 | 0.9624 | 0.9621 |
| mlp-h256_128_64-lr0.001-bs128 | 0.001 | (256, 128, 64) | 128 | 0.9706 | 0.9723 | 0.9721 |
| mlp-h128_64-lr0.001-bs128 | 0.001 | (128, 64) | 128 | 0.9724 | 0.9731 | 0.9730 |
| mlp-h64-lr0.001-bs128 | 0.001 | (64,) | 128 | 0.9684 | 0.9693 | 0.9692 |
