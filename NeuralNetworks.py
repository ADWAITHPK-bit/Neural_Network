"""
Neural Networks From Scratch: Forward and Backward — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  numerical_gradient ──
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x, dtype=float)
    for i in np.ndindex(x.shape):
        original = x[i]
        x[i] = original + eps
        f_plus = f(x)
        
        x[i] = original - eps
        f_minus = f(x)
        x[i] = original
        grad[i] = (f_plus - f_minus) / (2.0*eps)
    return grad
    pass

# ── Step 002  gradient_check ──
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    analytic_grad = np.asarray(analytic_grad, dtype=float)
    numeric_grad = np.asarray(numeric_grad, dtype=float)

    diff = np.abs(analytic_grad - numeric_grad)

    scale = np.maximum(np.maximum(analytic_grad, numeric_grad), tol)
    
    rel_error = diff / scale

    return float(np.max(rel_error))
    pass

# ── Step 003  make_dense ──
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim, out_dim)
    params = {
      "W": W,
      "b": b
    }
    def forward(x):
      W = params["W"]
      b = params["b"]

      y = x @ W + b
      cache = x
      return y, cache

    def backward(dout, cache):
      W = params["W"]
      b = params["b"]

      x = cache
      dx = dout @ W.T
      dW = x.T @ dout
      db = np.sum(dout,axis=0) # collapses rows 
      grads = {"W": dW, "b": db}
      return dx, grads

    dic = {
      "params": params,
      "forward": forward,
      "backward": backward
    }

    return dic

# ── Step 004  make_activation ──
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer."""

    if kind not in ["relu", "tanh"]:
        raise ValueError

    params = {}

    def forward(x):

        if kind == "relu":
            y = []

            for row in x:
                new_row = []

                for i in range(len(row)):
                    if row[i] > 0:
                        new_row.append(row[i])
                    else:
                        new_row.append(0.0)

                y.append(new_row)

            y = np.array(y)

        elif kind == "tanh":
            y = np.tanh(x)

        cache = x

        return y, cache

    def backward(dout, cache):

        if kind == "relu":
            der = []

            for row in cache:
                der_row = []

                for i in range(len(row)):
                    if row[i] > 0:
                        der_row.append(1.0)
                    else:
                        der_row.append(0.0)

                der.append(der_row)

            der = np.array(der)

        elif kind == "tanh":
            y = np.tanh(cache)
            der = 1 - y**2

        dx = der * dout

        return dx, params

    return {
        "params": params,
        "forward": forward,
        "backward": backward
    }

# ── Step 005  initialize_weights ──
# ── Step 005  initialize_weights ──
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    scheme = str(scheme).lower()
    
    if scheme in ['he', 'kaiming']:
        std = np.sqrt(2.0 / in_dim)
    elif scheme in ['xavier', 'glorot']:
        std = np.sqrt(2.0 / (in_dim + out_dim))
    elif scheme == 'normal':
        std = 0.01
    else:
        # Fallback to standard fan-in scaling
        std = np.sqrt(2.0 / in_dim)

    W = np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim))
    b = np.zeros(out_dim)

    return W, b

# ── Step 006  make_loss ──
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    if kind != "cross_entropy":
        raise ValueError

    def loss_fn(logits, labels):

        # Find maximum value in each row
        maxi = np.max(logits, axis=1, keepdims=True)

        # Shift logits for numerical stability
        y = logits - maxi

        # Exponentials
        y_exp = np.exp(y)

        # Sum exponentials row-wise
        y_exp_sum = np.sum(y_exp, axis=1, keepdims=True)

        # Log-sum-exp
        y_exp_sum_log = np.log(y_exp_sum) + maxi

        # Correct class logits
        correct_logits = logits[np.arange(len(labels)), labels]

        # Loss for each sample
        sample_loss = y_exp_sum_log.squeeze(1) - correct_logits

        # Mean loss
        loss = float(np.mean(sample_loss))

        # Softmax probabilities
        probs = y_exp / y_exp_sum

        # Gradient
        d_logits = probs.copy()

        # Subtract 1 from correct class
        d_logits[np.arange(len(labels)), labels] -= 1

        # Mean over batch
        d_logits /= len(labels)

        return loss, d_logits

    return loss_fn

# ── Step 007  make_sequential ──
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    def forward(x):
      current = x
      caches = []
      for layer in layers:
        current, cache = layer['forward'](current)
        caches.append(cache)
      
      return current, caches

    def backward(dout, caches):
      current_grad = dout
      grads_list = [None] * len(layers)
      for i in range(len(layers) -1, -1, -1):
        current_grad, grads = layers[i]['backward'](current_grad, caches[i])

        grads_list[i] = grads
        
      return current_grad, grads_list

    params = [layer['params'] for layer in layers]

    return {
    'forward': forward,
    'backward': backward,
    'params': params
    }
    pass

# ── Step 008  forward_backward ──
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # TODO: your approach here
    logits, caches = model['forward'](x)
    loss, dlogits = loss_fn(logits, y)
    _, params_grad = model['backward'](dlogits, caches)
    return loss, params_grad
    pass

# ── Step 009  make_optimizer ──
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    if kind != 'sgd':
      raise ValueError

    def helper(params,grads):
      if isinstance(params, dict):
        for key in params:
          helper(params[key], grads[key])

      elif isinstance(params, list):
        for i in range(len(params)):
          helper(params[i], grads[i])

      else:
        params -= lr * grads

    def step_grad(grads):
      helper(params, grads)

    return {
      'step': step_grad
    }

# ── Step 010  train_step ──
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    loss, param_grads = forward_backward(model, loss_fn, x_batch, y_batch)
    optim = optimizer['step'](param_grads)
    return loss
    pass

# ── Step 011  train ──
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    rng = np.random.RandomState(seed)
    history = []

    for epoch in range(epochs):
      indices = rng.permutation(len(x))
      epoch_losses = []

      for start in range(0, len(x), batch_size):
        batch_idx = indices[start:start + batch_size]
        x_batch = x[batch_idx]
        y_batch = y[batch_idx]
        loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)
        epoch_losses.append(loss)
      epoch_losses = np.array(epoch_losses)
      mean = np.mean(epoch_losses)
      history.append(mean)
    
    return history


    pass

# ── Step 012  design_network ──
# ── Step 012  design_network ──
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # ---------------------------------------------------------
    # 1. Synthesize concentric circular shells (strictly non-linear)
    # ---------------------------------------------------------
    samples_per_class = max(50, 300 // num_classes)
    x_list, y_list = [], []

    for c in range(num_classes):
        if input_dim == 1:
            signs = rng.choice([-1.0, 1.0], size=samples_per_class)
            radii = 0.4 * (c + 1) + rng.normal(0, 0.02, size=samples_per_class)
            pts = (signs * radii)[:, None]
        else:
            angles = rng.uniform(0.0, 2.0 * np.pi, size=samples_per_class)
            radii = 0.45 * (c + 1) + rng.normal(0, 0.03, size=samples_per_class)
            x0 = radii * np.cos(angles)
            x1 = radii * np.sin(angles)
            pts = np.column_stack([x0, x1])

            if input_dim > 2:
                extra = rng.normal(0, 0.01, size=(samples_per_class, input_dim - 2))
                pts = np.hstack([pts, extra])

        x_list.append(pts)
        y_list.append(np.full(samples_per_class, c, dtype=int))

    x = np.vstack(x_list).astype(float)
    y = np.concatenate(y_list).astype(int)

    # Deterministic shuffle
    shuffle_idx = rng.permutation(len(x))
    x = x[shuffle_idx]
    y = y[shuffle_idx]

    # ---------------------------------------------------------
    # 2. Build 3-Layer MLP with non-linear activations
    # ---------------------------------------------------------
    def init_fn(i, o):
        return initialize_weights(i, o, scheme='he')

    hidden_dim = 32
    layers = [
        make_dense(input_dim, hidden_dim, init_fn),
        make_activation('relu'),
        make_dense(hidden_dim, hidden_dim, init_fn),
        make_activation('relu'),
        make_dense(hidden_dim, num_classes, init_fn)
    ]
    model = make_sequential(layers)

    # Optimize with Cross-Entropy + SGD
    loss_fn = make_loss('cross_entropy')
    optimizer = make_optimizer(model['params'], lr=0.1, kind='sgd')

    train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        x=x,
        y=y,
        epochs=200,
        batch_size=32,
        seed=seed
    )

    # Report True Accuracy
    logits, _ = model['forward'](x)
    preds = np.argmax(logits, axis=1)
    accuracy = float(np.mean(preds == y))

    return model, {
        'accuracy': accuracy,
        'x': x,
        'y': y
    }

# ── Step 013  improve_generalization ──
# ── Step 013  improve_generalization ──
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline.

    Inputs:
      baseline_model_fn: zero-arg callable -> fresh untrained sequential model
        (dict with 'forward', 'backward', 'params') matching the data dims.
      x_train, y_train: training features (N, D) and int labels (N,).
      x_val, y_val: validation features (N_val, D) and int labels (N_val,).
      seed: int for deterministic training.

    Returns:
      dict with keys:
        'val_accuracy': float accuracy of the improved model on x_val/y_val
        'baseline_val_accuracy': float val accuracy of plain unregularized SGD
        'predictions': np.ndarray shape (N_val,) int preds from improved model
        'model': the trained improved model

    Required behavior:
      val_accuracy > baseline_val_accuracy
      predictions == argmax(model.forward(x_val), axis=1)
      val_accuracy == mean(predictions == y_val)
      predictions are non-constant (not a trivial single-class predictor)
    """
    loss_fn = make_loss('cross_entropy')

    # ---------------------------------------------------------
    # 1. Train Unregularized Baseline (Overfits noisy train labels)
    # ---------------------------------------------------------
    baseline_model = baseline_model_fn()
    baseline_opt = make_optimizer(baseline_model['params'], lr=0.1, kind='sgd')
    
    train(
        model=baseline_model,
        loss_fn=loss_fn,
        optimizer=baseline_opt,
        x=x_train,
        y=y_train,
        epochs=150,
        batch_size=32,
        seed=seed
    )
    
    base_logits, _ = baseline_model['forward'](x_val)
    base_preds = np.argmax(base_logits, axis=1)
    baseline_val_accuracy = float(np.mean(base_preds == y_val))

    # ---------------------------------------------------------
    # 2. Train Improved Model with L2 Regularization & Early Stopping
    # ---------------------------------------------------------
    improved_model = baseline_model_fn()
    improved_opt = make_optimizer(improved_model['params'], lr=0.08, kind='sgd')
    
    rng = np.random.RandomState(seed)
    weight_decay = 1e-2
    epochs = 150
    batch_size = 32

    best_val_acc = -1.0
    best_params = None

    for epoch in range(epochs):
        indices = rng.permutation(len(x_train))
        
        for start in range(0, len(x_train), batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]

            # Forward + Backward
            loss, param_grads = forward_backward(improved_model, loss_fn, xb, yb)

            # Apply L2 weight decay to weight gradients (dW += wd * W)
            for p_dict, g_dict in zip(improved_model['params'], param_grads):
                if 'W' in p_dict and 'W' in g_dict:
                    g_dict['W'] += weight_decay * p_dict['W']

            # Optimization step
            improved_opt['step'](param_grads)

        # Check validation accuracy after each epoch (Early Stopping Checkpoint)
        val_logits, _ = improved_model['forward'](x_val)
        val_preds = np.argmax(val_logits, axis=1)
        current_val_acc = float(np.mean(val_preds == y_val))

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_params = [
                {k: v.copy() for k, v in layer_p.items()}
                for layer_p in improved_model['params']
            ]

    # Restore best checkpoint
    if best_params is not None:
        for layer_p, best_layer_p in zip(improved_model['params'], best_params):
            for k in layer_p:
                np.copyto(layer_p[k], best_layer_p[k])


    # Final Evaluation

    final_logits, _ = improved_model['forward'](x_val)
    predictions = np.argmax(final_logits, axis=1)
    val_accuracy = float(np.mean(predictions == y_val))

    return {
        'val_accuracy': val_accuracy,
        'baseline_val_accuracy': baseline_val_accuracy,
        'predictions': predictions,
        'model': improved_model
    }

# ── Scaffold (runner) ──
"""End-to-end demo: NumPy neural net from scratch on a nonlinear dataset."""
import numpy as np


def _nonlinear_dataset(n_samples=256, seed=0, label_noise=0.0):
    """Binary labels on noisy concentric rings (not linearly separable)."""
    rng = np.random.RandomState(seed)
    half = n_samples // 2
    t = rng.uniform(0.0, 2.0 * np.pi, size=half)
    r0 = 0.45 + rng.normal(0.0, 0.06, size=half)
    r1 = 1.15 + rng.normal(0.0, 0.06, size=half)
    x0 = np.column_stack([r0 * np.cos(t), r0 * np.sin(t)])
    x1 = np.column_stack([r1 * np.cos(t), r1 * np.sin(t)])
    x = np.vstack([x0, x1]).astype(np.float64)
    y = np.array([0] * half + [1] * half, dtype=np.int64)
    if label_noise > 0.0:
        flip = rng.rand(n_samples) < float(label_noise)
        y = y.copy()
        y[flip] = 1 - y[flip]
    idx = rng.permutation(n_samples)
    return x[idx], y[idx]


def _fresh_mlp(in_dim, n_classes, hidden=32, seed=1):
    """Return a freshly initialized untrained sequential MLP."""
    np.random.seed(int(seed))

    def init_fn(i, o):
        return initialize_weights(i, o, scheme="he")

    layers = [
        make_dense(in_dim, hidden, init_fn),
        make_activation("relu"),
        make_dense(hidden, hidden, init_fn),
        make_activation("relu"),
        make_dense(hidden, n_classes, init_fn),
    ]
    return make_sequential(layers)


def main():
    np.random.seed(0)
    x, y = _nonlinear_dataset(256, seed=0, label_noise=0.0)
    n_train = 192
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]
    in_dim, n_classes = int(x.shape[1]), 2

    designed, design_metrics = design_network(in_dim, n_classes, seed=0)
    print("design_network_accuracy:", round(float(design_metrics["accuracy"]), 4))

    # Fresh untrained net on the rings data: one train_step must drop loss.
    model = _fresh_mlp(in_dim, n_classes, hidden=32, seed=1)
    loss_fn = make_loss("cross_entropy")
    xb, yb = x_train[:32], y_train[:32]
    init_loss, _ = forward_backward(model, loss_fn, xb, yb)
    print("initial_batch_loss:", float(init_loss))

    optimizer = make_optimizer(model["params"], lr=0.1, kind="sgd")
    pre_loss = train_step(model, loss_fn, optimizer, xb, yb)
    post_loss, _ = forward_backward(model, loss_fn, xb, yb)
    print("train_step_loss:", float(pre_loss))
    print("after_train_step_loss:", float(post_loss))

    history = train(
        model, loss_fn, optimizer, x_train, y_train,
        epochs=50, batch_size=32, seed=0,
    )
    if isinstance(history, (list, tuple, np.ndarray)) and len(history) > 0:
        print("overfit_loss_start:", float(history[0]))
        print("overfit_loss_end:", float(history[-1]))
    else:
        print("train_history:", history)

    # Regularization demo: wide net + noisy train labels so the unregularized
    # baseline cannot already sit at 100% val accuracy.
    x_noisy, y_noisy = _nonlinear_dataset(192, seed=2, label_noise=0.22)
    x_hold, y_hold = _nonlinear_dataset(64, seed=3, label_noise=0.0)

    def baseline_model_fn():
        return _fresh_mlp(in_dim, n_classes, hidden=56, seed=12345)

    gen_result = improve_generalization(
        baseline_model_fn, x_noisy, y_noisy, x_hold, y_hold, seed=0,
    )
    print("baseline_val_accuracy:", round(float(gen_result["baseline_val_accuracy"]), 4))
    print("improved_val_accuracy:", round(float(gen_result["val_accuracy"]), 4))


if __name__ == "__main__":
    main()
