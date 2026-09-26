"""
Denoising Diffusion (DDPM) from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  linear_beta_schedule ──
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    if T == 1:
        return torch.tensor([beta_start], dtype=torch.float32)

    betas = []
    for i in range(T):
        betas.append(beta_start + ((i)/(T-1))*(beta_end - beta_start))
    betas = torch.tensor(betas)
    return betas
    # betas = torch.linspace(beta_start,beta_end,T)
    # return betas
    pass

# ── Step 002  alphas_from_betas ──
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1 - betas
    pass

# ── Step 003  cumprod_alphas ──
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas,dim=0)
    pass

# ── Step 004  extract_into_batch ──
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    a = a.gather(0,t.long())
    a = a.reshape((-1,1,1,1))
    return a
    pass

# ── Step 005  q_sample ──
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    bar_alpha_t = extract_into_batch(alphas_cumprod,t,x0)
    x_t = torch.sqrt(bar_alpha_t)*x0 + torch.sqrt(1 - bar_alpha_t)*noise
    return x_t
    pass

# ── Step 006  build_diffusion_schedule ──
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = 1 - sqrt_alphas_cumprod
    return {
    "T": T,
    "betas": betas,
    "alphas": alphas,
    "alphas_cumprod": alphas_cumprod,
    "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
    "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }
    pass

# ── Step 007  noise_prediction_loss ──
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    n = len(noise)
    return torch.mean((noise - noise_pred)**2)
    pass

# ── Step 008  diffusion_training_loss ──
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    x_t = q_sample(x0, t, noise,alphas_cumprod)
    noise_pred = model(x_t, t)
    mse = noise_prediction_loss(noise_pred, noise)
    return mse
    pass

# ── Step 009  timestep_embedding ──
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    half = dim // 2

    if half == 1:
        exponents = torch.zeros(1, device=t.device)
    else:
        exponents = torch.arange(half, device=t.device) / (half - 1)

    freqs = 10000**exponents

    angles = t.float().unsqueeze(1) / freqs.unsqueeze(0)

    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return emb

    pass

# ── Step 010  init_tiny_unet ──
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)

    conv_in_w = torch.randn(hidden, in_ch, 3, 3) * 0.02
    conv_in_w.requires_grad_()

    conv_in_b = torch.zeros(hidden)
    conv_in_b.requires_grad_()

    time_mlp_w = torch.randn(hidden,time_dim) * 0.02
    time_mlp_w.requires_grad_()

    time_mlp_b = torch.zeros(hidden)
    time_mlp_b.requires_grad_()

    conv_mid_w = torch.randn(hidden, hidden, 3, 3) * 0.02
    conv_mid_w.requires_grad_()

    conv_mid_b = torch.zeros(hidden)
    conv_mid_b.requires_grad_()

    conv_out_w = torch.randn(in_ch, hidden, 3, 3) * 0.02
    conv_out_w.requires_grad_()

    conv_out_b = torch.zeros(in_ch)
    conv_out_b.requires_grad_()

    return{
        "conv_in_w": conv_in_w,
        "conv_in_b": conv_in_b,
        "time_mlp_w": time_mlp_w,
        "time_mlp_b": time_mlp_b,
        "conv_mid_w": conv_mid_w,
        "conv_mid_b": conv_mid_b,
        "conv_out_w": conv_out_w,
        "conv_out_b": conv_out_b
    }
    pass

# ── Step 011  tiny_unet_forward ──
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise
    conv_in_w = params["conv_in_w"]
    conv_in_b = params["conv_in_b"]
    time_mlp_w = params["time_mlp_w"]
    time_mlp_b = params["time_mlp_b"]
    conv_mid_w = params["conv_mid_w"]
    conv_mid_b = params["conv_mid_b"]
    conv_out_w = params["conv_out_w"]
    conv_out_b = params["conv_out_b"]

    h = F.conv2d(x, conv_in_w, conv_in_b, padding=1)
    temb = timestep_embedding(t, params["time_mlp_w"].shape[1])
    temb = F.relu(F.linear(temb, time_mlp_w, time_mlp_b))

    temb = temb[:, :,None, None]
    h = h + temb

    h = F.relu(F.conv2d(h, conv_mid_w, conv_mid_b, padding=1))
    return F.conv2d(h, conv_out_w, conv_out_b, padding=1)
    pass

# ── Step 012  make_blob_dataset ──
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    torch.manual_seed(seed)
    radius = size // 4
    temp = torch.zeros(n,1,size,size)

    for i in range(n):
        center = torch.randint(radius, size-radius,(2,))
        cy = center[0]
        cx = center[1]

        for y in range(size):
            for x in range(size):
                if (y - cy)**2 + (x - cx)**2 <= radius**2:
                    temp[i,0,y,x] = 1
                else:
                    temp[i,0,y,x] = 0

    return temp.float()

            


    pass

# ── Step 013  ddpm_train_step ──
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # TODO: sample t,noise -> loss -> SGD on params
    torch.manual_seed(seed)
    batch_size = x0.shape[0]

    T = schedule["T"]
    alphas_cumprod = schedule["alphas_cumprod"]

    t = torch.randint(0, T,(batch_size,))
    noise = torch.randn_like(x0)

    loss = diffusion_training_loss(lambda x,t: tiny_unet_forward(x,t,params),x0, t, noise, alphas_cumprod)

    loss.backward()

    new_params = {}

    for name, p in params.items():
        if p.grad is not None:
            new_params[name] = (p - lr*p.grad).detach().requires_grad_(True)
        else:
            new_params[name] = p.detach().clone().requires_grad_(True)
    return new_params, float(loss) 
    pass

# ── Step 014  train_ddpm ──
# import torch
# import torch.nn.functional as F

# def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
#     # TODO: minibatch SGD training loop
#     history = []
#     for step in range(num_steps):
#         torch.manual_seed(seed + step)
#         n = len(dataset)
#         indices = torch.randint(0, n, (batch_size,))
#         x0 = dataset[indices]
        
#         result = ddpm_train_step(
#             params, x0, schedule,
#             lr, seed + step
#         )
#         loss = result[1]
#         param = result[0]

#         history.append(loss)

#     return params, history
        
#     pass

def train_ddpm(dataset, params: dict, schedule: dict,
               num_steps: int = 50, batch_size: int = 16,
               lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:

    history = []

    for step in range(num_steps):
        torch.manual_seed(seed + step)

        n = len(dataset)
        indices = torch.randint(0, n, (batch_size,))
        x0 = dataset[indices]

        result = ddpm_train_step(
            params, x0, schedule,
            lr, seed + step
        )

        params = result[0]
        loss = result[1]

        history.append(loss)

    return params, history

# ── Step 015  predict_x0_from_eps ──
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    alphas_cumprod = extract_into_batch(alphas_cumprod, t, x_t)
    x0_hat = (x_t - torch.sqrt(1 - alphas_cumprod) * eps) / torch.sqrt(alphas_cumprod)
    return x0_hat
    pass

# ── Step 016  ddpm_p_mean_variance ──
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # TODO: return (posterior_mean, variance, x0_hat)
    alphas = schedule["alphas"]
    alphas_cumprod = schedule["alphas_cumprod"]
    betas = schedule["betas"]
    
    x0_hat = predict_x0_from_eps(x_t, t, eps, alphas_cumprod)
    x0_hat = torch.clamp(x0_hat, -1, 1)

    alpha_t = alphas[t].view(-1, 1, 1, 1)
    beta_t = betas[t].view(-1, 1, 1, 1)
    alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1, 1)
# torch.where(condition, value_if_true, value_if_false)
    alpha_bar_prev = torch.where(
        t == 0,
        torch.ones_like(t, dtype=alphas_cumprod.dtype),
        alphas_cumprod[t - 1]
    ).view(-1, 1, 1, 1)

    mean = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t) * x0_hat + (torch.sqrt(alpha_t)*(1 - alpha_bar_prev) / (1 - alpha_bar_t))*x_t
    variance = beta_t
    return mean, variance, x0_hat
    pass

# ── Step 017  ddpm_p_sample ──
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # TODO: one reverse step x_t -> x_{t-1}
    eps = tiny_unet_forward(x_t, t, params)
    mean, var, _ = ddpm_p_mean_variance(x_t, t, eps, schedule)

    if noise is None:
        noise = torch.randn_like(x_t)

    # if t.item() == 0:
    #     x_prev = mean
    # else:
    #     x_prev = mean + torch.sqrt(var) * noise

    x_prev = torch.where(
        (t == 0).reshape(-1, 1, 1, 1),
        mean,
        mean + torch.sqrt(var) * noise
    )
    return x_prev
    pass

# ── Step 018  ddpm_sample_loop ──
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    # TODO: ancestral sampling from pure noise to x0
    # x = torch.randn(shape)
    torch.manual_seed(seed)
    x = torch.randn(shape)
    B = shape[0]
    T = len(schedule["betas"])
    
    for t in range(T-1,-1,-1):
        t_batch = torch.full((B,), t, dtype=torch.long)
        x = ddpm_p_sample(x, t_batch, params, schedule)
    return x
    pass

# ── Step 019  sample_quality_mse ──
# import torch
# import torch.nn.functional as F

# def sample_quality_mse(samples, dataset) -> float:
#     # TODO: mean over samples of min MSE to any dataset image
#     samples_flat = samples.reshape(samples.shape[0], -1)
#     dataset_flat = dataset.reshape(dataset.shape[0], -1)

#     temp = []
#     # n = len(samples_flat)
#     for i in range(len(samples_flat)):
#         mse_list = []
#         for j in range(len(dataset_flat)):
#             mse = torch.mean((samples_flat[i] - dataset_flat[j]))**2
#             mse_list.append(mse.item())

#         min_mse = min(mse_list)
#         temp.append(min_mse)
#     mean_mse = sum(temp) / len(temp)
#     return float(mean_mse)
#     pass

def sample_quality_mse(samples, dataset) -> float:
    samples_flat = samples.reshape(samples.shape[0], -1)
    dataset_flat = dataset.reshape(dataset.shape[0], -1)

    temp = []

    for i in range(len(samples_flat)):
        min_mse = None

        for j in range(len(dataset_flat)):
            mse = torch.mean(
                (samples_flat[i] - dataset_flat[j]) ** 2
            )

            if min_mse is None or mse < min_mse:
                min_mse = mse

        temp.append(min_mse.item())

    mean_mse = sum(temp) / len(temp)

    return float(mean_mse)

# ── Step 020  ddpm_experiment ──
# import torch
# import torch.nn.functional as F

# def ddpm_experiment(n_data: int = 64, size: int = 8, T: int = 20, hidden: int = 16, num_steps: int = 40, batch_size: int = 16, lr: float = 5e-2, n_samples: int = 8, seed: int = 0) -> dict:
#     # TODO: data -> train -> sample -> metrics
#     dataset = make_blob_dataset(n_data, size, seed)
#     schedule = build_diffusion_schedule(T)
    
#     params = init_tiny_unet(1, hidden, time_dim=hidden, seed=seed)
    
#     params, history = train_ddpm(dataset, params, schedule
#                                 , num_steps,
#                                 batch_size, lr, seed)
    
#     samples = ddpm_sample_loop(params, schedule,
#                                 (n_samples,1, size, size),
#                                 seed=seed+1)
    
#     torch.manual_seed(seed + 2)
#     noise = torch.randn((n_samples, 1, size, size))

#     sample_mse = sample_quality_mse(samples, dataset)
#     noise_mse = sample_quality_mse(noise, dataset)

#     return {
#         "train_losses": [float(x) for x in history],
#         "final_loss": float(history[-1]),
#         "sample_mse": float(sample_mse),
#         "noise_mse": float(noise_mse),
#         "improvement": float(noise_mse - sample_mse)
#     }
#     pass
import torch
import torch.nn.functional as F

def ddpm_experiment(
    n_data: int = 64,
    size: int = 8,
    T: int = 20,
    hidden: int = 16,
    num_steps: int = 40,
    batch_size: int = 16,
    lr: float = 5e-2,
    n_samples: int = 8,
    seed: int = 0
) -> dict:

    dataset = make_blob_dataset(n_data, size, seed)
    schedule = build_diffusion_schedule(T)

    params = init_tiny_unet(1, hidden, time_dim=hidden, seed=seed)

    params, history = train_ddpm(
        dataset,
        params,
        schedule,
        num_steps,
        batch_size,
        lr,
        seed
    )

    samples = ddpm_sample_loop(
        params,
        schedule,
        (n_samples, 1, size, size),
        seed=seed + 1
    )

    torch.manual_seed(seed + 2)
    noise = torch.randn((n_samples, 1, size, size))

    sample_mse = sample_quality_mse(samples, dataset)
    noise_mse = sample_quality_mse(noise, dataset)

    return {
        "train_losses": [float(x) for x in history],
        "final_loss": float(history[-1]),
        "sample_mse": float(sample_mse),
        "noise_mse": float(noise_mse),
        "improvement": float(noise_mse - sample_mse)
    }

# ── Scaffold (runner) ──
"""End-to-end demo: train a tiny DDPM on synthetic blob images and sample new ones.

Story: pure Gaussian noise is unstructured (high nearest-neighbor MSE to the data).
After a short training run the reverse process produces images much closer to the
bright-disk manifold — visible both as a drop in training loss and as a lower
sample_quality_mse than the noise baseline.
"""
# Imports live here too: /assemble concatenates solutions FIRST, then this
# scaffolding. Names like F are resolved at call time inside main(), so these
# imports cover user solutions that used F.conv2d / torch.* without importing.
import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)
    result = ddpm_experiment(
        n_data=64,
        size=8,
        T=20,
        hidden=16,
        num_steps=60,
        batch_size=16,
        lr=5e-2,
        n_samples=8,
        seed=0,
    )
    print("steps:", len(result["train_losses"]))
    print(f"loss: {result['train_losses'][0]:.4f} -> {result['final_loss']:.4f}")
    print(f"noise baseline MSE:  {result['noise_mse']:.4f}")
    print(f"trained sample MSE:  {result['sample_mse']:.4f}")
    print(f"improvement (noise - sample): {result['improvement']:.4f}")


if __name__ == "__main__":
    main()
