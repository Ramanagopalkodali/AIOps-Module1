import time

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# MLflow configuration
# ============================================================

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "mnist-mlp"


# ============================================================
# Load MNIST once
# ============================================================

def load_mnist():

    print("Loading MNIST dataset...")

    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="auto"
    )

    X = X.astype(np.float32)
    y = y.astype(int)

    # Convert pixels from 0-255 to 0-1
    X = X / 255.0

    print("Dataset loaded.")
    print("Shape:", X.shape)

    return X, y


# ============================================================
# Train one experiment
# ============================================================

def train_and_log(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    learning_rate,
    hidden_layer_sizes,
    batch_size=128,
    max_iter=30
):

    architecture_name = "_".join(
        str(x) for x in hidden_layer_sizes
    )

    run_name = (
        f"mlp-h{architecture_name}"
        f"-lr{learning_rate}"
        f"-bs{batch_size}"
    )

    print("\n" + "=" * 70)
    print("Running:", run_name)
    print("=" * 70)

    with mlflow.start_run(run_name=run_name) as run:

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            "MLPClassifier"
        )

        mlflow.log_param(
            "dataset",
            "MNIST"
        )

        mlflow.log_param(
            "hidden_layer_sizes",
            str(hidden_layer_sizes)
        )

        mlflow.log_param(
            "learning_rate_init",
            learning_rate
        )

        mlflow.log_param(
            "batch_size",
            batch_size
        )

        mlflow.log_param(
            "alpha",
            0.0001
        )

        mlflow.log_param(
            "activation",
            "relu"
        )

        mlflow.log_param(
            "solver",
            "adam"
        )

        mlflow.log_param(
            "max_iter",
            max_iter
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.set_tag(
            "question",
            "Q2-MLflow-MNIST-MLP"
        )


        # ----------------------------------------------------
        # Create MLP
        # ----------------------------------------------------

        model = MLPClassifier(

            hidden_layer_sizes=hidden_layer_sizes,

            learning_rate_init=learning_rate,

            batch_size=batch_size,

            alpha=0.0001,

            activation="relu",

            solver="adam",

            max_iter=max_iter,

            early_stopping=True,

            validation_fraction=0.15,

            n_iter_no_change=30,

            random_state=42
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        train_time = time.time() - start_time


        # ----------------------------------------------------
        # Log training-loss curve
        # ----------------------------------------------------

        for epoch, loss in enumerate(
            model.loss_curve_
        ):

            mlflow.log_metric(
                "train_loss",
                loss,
                step=epoch
            )


        # ----------------------------------------------------
        # Log validation-accuracy curve
        # ----------------------------------------------------

        if hasattr(
            model,
            "validation_scores_"
        ):

            for epoch, val_acc_epoch in enumerate(
                model.validation_scores_
            ):

                mlflow.log_metric(
                    "val_accuracy_epoch",
                    val_acc_epoch,
                    step=epoch
                )


        # ----------------------------------------------------
        # Final validation metrics
        # ----------------------------------------------------

        val_predictions = model.predict(
            X_val
        )

        val_accuracy = accuracy_score(
            y_val,
            val_predictions
        )


        # ----------------------------------------------------
        # Final test metrics
        # ----------------------------------------------------

        test_predictions = model.predict(
            X_test
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        test_f1 = f1_score(
            y_test,
            test_predictions,
            average="macro"
        )


        # ----------------------------------------------------
        # Log final metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "val_accuracy",
            val_accuracy
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "test_f1_macro",
            test_f1
        )

        mlflow.log_metric(
            "n_epochs_run",
            len(model.loss_curve_)
        )

        mlflow.log_metric(
            "train_time_sec",
            train_time
        )


        # ----------------------------------------------------
        # Log model
        # Compatible with your MLflow installation
        # ----------------------------------------------------

        signature = infer_signature(
            X_train[:5],
            model.predict(X_train[:5])
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5]
        )


        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"run_id       = {run.info.run_id}"
        )

        print(
            f"architecture = {hidden_layer_sizes}"
        )

        print(
            f"learning_rate = {learning_rate}"
        )

        print(
            f"batch_size    = {batch_size}"
        )

        print(
            f"val_accuracy  = {val_accuracy:.4f}"
        )

        print(
            f"test_accuracy = {test_accuracy:.4f}"
        )

        print(
            f"test_f1       = {test_f1:.4f}"
        )

        print(
            f"epochs        = {len(model.loss_curve_)}"
        )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Connect to MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


    # --------------------------------------------------------
    # Load MNIST ONCE
    # --------------------------------------------------------

    X, y = load_mnist()


    # --------------------------------------------------------
    # Train / validation / test split
    # --------------------------------------------------------

    X_train_full, X_test, y_train_full, y_test = train_test_split(

        X,
        y,

        test_size=0.15,

        random_state=42,

        stratify=y
    )


    X_train, X_val, y_train, y_val = train_test_split(

        X_train_full,
        y_train_full,

        test_size=0.15,

        random_state=42,

        stratify=y_train_full
    )


    # --------------------------------------------------------
    # Scale data
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )


    # ========================================================
    # SIX REQUIRED EXPERIMENTS
    #
    # Hyperparameter 1:
    #   learning_rate_init
    #
    # Hyperparameter 2:
    #   hidden_layer_sizes
    #
    # 2 × 3 = 6 runs
    # ========================================================

    experiments = [

        (0.001, (64,)),

        (0.001, (128, 64)),

        (0.001, (256, 128, 64)),

        (0.01, (64,)),

        (0.01, (128, 64)),

        (0.01, (256, 128, 64)),
    ]


    # --------------------------------------------------------
    # Run all six experiments
    # --------------------------------------------------------

    for learning_rate, architecture in experiments:

        train_and_log(

            X_train=X_train,

            X_val=X_val,

            X_test=X_test,

            y_train=y_train,

            y_val=y_val,

            y_test=y_test,

            learning_rate=learning_rate,

            hidden_layer_sizes=architecture,

            batch_size=128,

            max_iter=30
        )


    print("\n")
    print("=" * 70)
    print("ALL SIX EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(
        f"MLflow experiment: {EXPERIMENT_NAME}"
    )
    print(
        "Open http://localhost:5000"
    )


if __name__ == "__main__":

    main()

