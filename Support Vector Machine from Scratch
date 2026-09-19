"""
Support Vector Machine from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  standardize_features ──
import numpy as np

def standardize_features(x):
    # TODO: rescale each column of x to have mean 0 and std 1 (leave zero-std columns alone).
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    temp = []
    return np.where(std == 0, x - mean, (x - mean) / std)
    pass

# ── Step 002  initialize_parameters ──
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    w = np.zeros(n_features)
    b = 0.0
    result = {
        "w": w,
        "b": b
    }
    return result
    pass

# ── Step 003  compute_scores ──
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    w = params["w"]
    b = params["b"]

    result = x @ w + b
    
    return result
    pass

# ── Step 004  predict_from_scores ──
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    out = np.where(scores>=0, 1, -1)
    return out
    pass

# ── Step 005  hinge_loss_example ──
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    m = 1 - (y*score)
    if m >= 0:
        return m
    else:
        return 0.0
    pass

# ── Step 006  svm_objective ──
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    w = params["w"]
    b = params["b"]

    score = compute_scores(x, params)
    # hinge_loss = hinge_loss_example(score, y)

    hinge_loss = np.maximum(0.0, 1.0 - y*score)
    hinge_loss = np.mean(hinge_loss)
    hinge_loss = hinge_loss + reg_lambda*np.dot(w,w)

    return float(hinge_loss)
    pass

# ── Step 007  compute_gradients ──
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # TODO: compute the gradient of the SVM objective wrt params['w'] and params['b'].
    score = compute_scores(x, params)
    m = 1 - y*score
    mask = m > 0

    n = x.shape[0]

    # Hinge-loss gradient
    dw = -np.sum(
        y[mask, np.newaxis] * x[mask],
        axis=0
    ) / n

    db = -np.sum(y[mask]) / n

    # L2 regularization gradient
    dw += 2 * reg_lambda * params["w"]

    return {
        "dw": dw,
        "db": float(db)
    }
    pass

# ── Step 008  apply_update ──
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    w = params["w"]
    b = params["b"]

    dw = grads["dw"]
    db = grads["db"]

    new_weights = w - learning_rate*dw
    new_bias = b - learning_rate*db

    return {
        "w": new_weights,
        "b": new_bias
    }
    pass

# ── Step 009  train_svm ──
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    # TODO: fit a linear SVM by repeatedly updating parameters over n_epochs passes.
    # params = initialize_parameters(n_features)
    # w = params["w"]
    # b = params["b"]

    params = initialize_parameters(x.shape[1])
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    return params
    pass

# ── Step 010  predict_labels ──
import numpy as np

def predict_labels(x, params):
    # TODO: return an array of {-1, +1} labels, one per row of x, using params['w'] and params['b'].
    score = compute_scores(x, params)
    result = predict_from_scores(score)
    return np.array(result)
    pass

# ── Step 011  accuracy_score ──
import numpy as np

def accuracy_score(y_pred, y_true):
    # TODO: return the fraction of positions where y_pred equals y_true.
    result = [y_pred==y_true]
    mean = np.mean(result)
    return float(mean)
    pass

# ── Scaffold (runner) ──
"""Demo scaffold: train a linear SVM from scratch on synthetic 2D data."""

import numpy as np


def make_toy_dataset(n_per_class=50):
    rng = np.random.default_rng(0)
    x_pos = rng.normal(loc=(2.0, 2.0), scale=0.8, size=(n_per_class, 2))
    x_neg = rng.normal(loc=(-2.0, -2.0), scale=0.8, size=(n_per_class, 2))
    x = np.vstack([x_pos, x_neg])
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    np.random.seed(0)

    # 1. Data prep
    x_raw, y = make_toy_dataset(n_per_class=60)
    x = standardize_features(x_raw)
    print("Data shapes: x =", x.shape, " y =", y.shape)
    print("First standardized rows:\n", np.round(x[:3], 3))
    print("Feature means ~0:", np.round(x.mean(axis=0), 3),
          " std ~1:", np.round(x.std(axis=0), 3))

    # 2. Forward pass with freshly initialized params
    n_features = x.shape[1]
    init_params = initialize_parameters(n_features)
    print("\nInitial params:", init_params)

    init_scores = compute_scores(x[:5], init_params)
    print("Initial scores (first 5):", np.round(init_scores, 4))
    print("Initial predictions:", predict_from_scores(init_scores))
    print("Hinge loss on example 0:",
          round(float(hinge_loss_example(float(init_scores[0]), float(y[0]))), 4))

    reg_lambda = 0.01
    init_obj = svm_objective(x, y, init_params, reg_lambda)
    print("Initial SVM objective:", round(float(init_obj), 4))

    # 3. A single manual gradient/update step (sanity check)
    grads = compute_gradients(x, y, init_params, reg_lambda)
    stepped_params = apply_update(init_params, grads, learning_rate=0.1)
    stepped_obj = svm_objective(x, y, stepped_params, reg_lambda)
    print("Objective after one manual step:", round(float(stepped_obj), 4))

    # 4. Full training loop
    trained_params = train_svm(
        x, y,
        learning_rate=0.05,
        reg_lambda=reg_lambda,
        n_epochs=200,
    )
    print("\nTrained params:", trained_params)
    print("Final SVM objective:",
          round(float(svm_objective(x, y, trained_params, reg_lambda)), 4))

    # 5. Predict & evaluate
    y_pred = predict_labels(x, trained_params)
    acc = float(np.mean(np.asarray(y_pred) == np.asarray(y)))
    print("Training accuracy:", round(float(acc), 4))
    print("First 10 predictions:", y_pred[:10].astype(int))
    print("First 10 true labels:", y[:10].astype(int))


if __name__ == "__main__":
    main()
