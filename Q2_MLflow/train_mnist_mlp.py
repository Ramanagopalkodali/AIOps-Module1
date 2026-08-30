import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. MLflow configuration
# --------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist-mlp-q2")


# --------------------------------------------------
# 2. Load MNIST
# --------------------------------------------------

print("Loading MNIST dataset...")

X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False,
    parser="auto"
)

X = X.astype(np.float32)
y = y.astype(np.int64)

print("Dataset loaded.")
print("Shape:", X.shape)


# --------------------------------------------------
# 3. Use a smaller subset for faster experiments
# --------------------------------------------------

X = X[:12000]
y = y[:12000]

print("Using subset:", X.shape)


# --------------------------------------------------
# 4. Train / validation split
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Scale MNIST pixels
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# --------------------------------------------------
# 6. Function for one MLflow experiment
# --------------------------------------------------

def train_and_log(learning_rate, batch_size, hidden_layer_size):

    run_name = (
        f"mlp-lr-{learning_rate}-batch-{batch_size}"
    )

    with mlflow.start_run(run_name=run_name):

        # ------------------------------
        # Log hyperparameters
        # ------------------------------

        mlflow.log_param(
            "learning_rate_init",
            learning_rate
        )

        mlflow.log_param(
            "batch_size",
            batch_size
        )

        mlflow.log_param(
            "hidden_layer_size",
            hidden_layer_size
        )

        mlflow.log_param(
            "max_iter",
            20
        )

        mlflow.log_param(
            "random_state",
            42
        )


        # ------------------------------
        # Create MLP
        # ------------------------------

        model = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            learning_rate_init=learning_rate,
            batch_size=batch_size,
            max_iter=20,
            random_state=42,
            early_stopping=False,
            verbose=False
        )


        # ------------------------------
        # Train model
        # ------------------------------

        print(
            f"\nTraining: "
            f"learning_rate={learning_rate}, "
            f"batch_size={batch_size}"
        )

        model.fit(
            X_train,
            y_train
        )


        # ------------------------------
        # Predictions
        # ------------------------------

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)


        # ------------------------------
        # Metrics
        # ------------------------------

        train_accuracy = accuracy_score(
            y_train,
            train_predictions
        )

        val_accuracy = accuracy_score(
            y_val,
            val_predictions
        )

        train_loss = model.loss_


        # ------------------------------
        # Log metrics
        # ------------------------------

        mlflow.log_metric(
            "train_loss",
            train_loss
        )

        mlflow.log_metric(
            "train_accuracy",
            train_accuracy
        )

        mlflow.log_metric(
            "val_accuracy",
            val_accuracy
        )


        # ------------------------------
        # Log model
        # ------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )


        # ------------------------------
        # Print result
        # ------------------------------

        print(
            f"train_loss={train_loss:.4f} | "
            f"train_accuracy={train_accuracy:.4f} | "
            f"val_accuracy={val_accuracy:.4f}"
        )


# --------------------------------------------------
# 7. SIX EXPERIMENTS
# --------------------------------------------------

experiments = [

    # Learning rate = 0.001
    (0.001, 32, 128),
    (0.001, 64, 128),
    (0.001, 128, 128),

    # Learning rate = 0.01
    (0.01, 32, 128),
    (0.01, 64, 128),
    (0.01, 128, 128),
]


for learning_rate, batch_size, hidden_layer_size in experiments:

    train_and_log(
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_layer_size=hidden_layer_size
    )


print("\nAll six experiments completed.")
