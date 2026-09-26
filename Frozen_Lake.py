"""
Q-Learning on FrozenLake from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  init_q_table ──
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # TODO: build a 2D float64 numpy array of zeros sized by states and actions.
    return np.zeros((num_states, num_actions))
    pass

# ── Step 002  max_q_value ──
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    # TODO: index the row for `state` and return its maximum value
    max = q_table[state][0]
    for num in q_table[state]:
        if max < num:
            max = num

    return max
    pass
# import numpy as np

# def max_q_value(q_table, state):
#     """Return the maximum Q value across all actions for the given state."""
#     return np.max(q_table[state])

# ── Step 003  greedy_action ──
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    # TODO: return argmax over the action axis for this state's Q values
    index = 0
    max = q_table[state][0]
    for i,num in enumerate(q_table[state]):
        if max < num:
            max = num
            index = i
        
    return index
    pass

# import numpy as np

# def greedy_action(q_table, state):
#     return np.argmax(q_table[state])

# ── Step 004  sample_random_action ──
def sample_random_action(action_space):
    # TODO: draw a uniformly random action from the given Gymnasium action space
    return int(action_space.sample())
    pass

# ── Step 005  should_explore ──
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # TODO: draw a uniform sample from rng and compare it to epsilon
    if rng.random() > epsilon:
        return False
    else:
        return True
    pass

# ── Step 006  epsilon_greedy_action ──
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # TODO: with prob epsilon explore via action_space, else take greedy action
    if should_explore(epsilon,rng):
        return sample_random_action(action_space)
    else:
        return greedy_action(q_table, state)

    pass

# ── Step 007  decay_epsilon ──
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # TODO: return max(min_epsilon, epsilon * decay_rate)
    return max(min_epsilon, epsilon*decay_rate)
    pass

# ── Step 008  td_target ──
def td_target(reward, gamma, q_table, next_state, done):
    # TODO: compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done.
    if done:
        return reward
    else:
        return reward + gamma * max(q_table[next_state])
    pass

# ── Step 009  td_error ──
def td_error(target, q_table, state, action):
    # TODO: return the TD error: target minus current Q(state, action)
    return target - (q_table[state][action])
    pass

# ── Step 010  q_learning_update ──
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state, done)
    q_table[state][action] += alpha * (target - q_table[state][action])
    return q_table[state][action]  
    pass

# ── Step 011  interaction_step ──
# def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
#     # TODO: select epsilon-greedy action, step env, apply Q-learning update, return (next_state, reward, done)
#     action = epsilon_greedy_action(q_table, state, epsilon,env.action_space, rng)
    
#     # Take actions
#     next_state, reward, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated

#     #Update q_table
#     update = q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)
    
#     return (int(next_state), float(reward), done)
#     pass
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):

    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)

    return int(next_state), float(reward), done

# ── Step 012  run_training_episode ──
# def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
#     # TODO: reset env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    
#     state, i = env.reset()
#     su = 0.0
#     for i in range(max_steps):
#         next_state, reward, done = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
#         su += reward
#         state = next_state

#         # if done:
#         #     break

#     return su
#     pass
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):

    state, _ = env.reset()
    su = 0.0

    for _ in range(max_steps):
        next_state, reward, done = interaction_step(
            env, q_table, state, epsilon, alpha, gamma, rng
        )

        su += reward
        state = next_state

        if done:
            break

    return su

# ── Step 013  train_q_learning ──
# import numpy as np

# def train_q_learning(env, num_episodes,
#                      alpha=0.1,
#                      gamma=0.99,
#                      epsilon_start=1.0,
#                      epsilon_min=0.05,
#                      epsilon_decay=0.995,
#                      seed=0,
#                      max_steps=200):

#     rng = np.random.default_rng(seed)

#     # Required for reproducibility
#     env.reset(seed=seed)
#     env.action_space.seed(seed)

#     q_table = np.zeros((env.observation_space.n,
#                         env.action_space.n))

#     episode_returns = []
#     epsilon = epsilon_start

#     for i in range(num_episodes):
#         reward = run_training_episode(
#             env,
#             q_table,
#             epsilon,
#             alpha,
#             gamma,
#             rng,
#             max_steps
#         )

#         episode_returns.append(float(reward))
#         epsilon = max(epsilon_min,
#                       epsilon * epsilon_decay)

#     return q_table, episode_returns

import numpy as np

def train_q_learning(env, num_episodes,
                     alpha=0.1,
                     gamma=0.99,
                     epsilon_start=1.0,
                     epsilon_min=0.05,
                     epsilon_decay=0.995,
                     seed=0,
                     max_steps=200):

    rng = np.random.default_rng(seed)

    # reproducibility
    env.reset(seed=seed)
    env.action_space.seed(seed)

    q_table = np.zeros((env.observation_space.n,
                        env.action_space.n))

    episode_returns = []
    epsilon = epsilon_start

    for _ in range(num_episodes):

        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):

            action = epsilon_greedy_action(
                q_table, state, epsilon, env.action_space, rng
            )

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            best_next = np.max(q_table[next_state])
            td_target = reward + gamma * best_next * (0.0 if done else 1.0)
            q_table[state, action] += alpha * (td_target - q_table[state, action])

            state = next_state
            total_reward += reward

            if done:
                break

        episode_returns.append(total_reward)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, episode_returns

# ── Step 014  extract_greedy_policy ──
def extract_greedy_policy(q_table):
    # TODO: return a 1D int64 array mapping each state to its best (argmax) action.
    return np.argmax(q_table, axis = 1)
    pass

# ── Step 015  run_greedy_episode ──
def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
    # TODO: reset env, follow policy[state] each step, return bool(success)
    state,_ = env.reset(seed=seed)
    
    for i in range(max_steps):
        action = policy[state]
        state, reward, terminated, turnicated, _ = env.step(action)

        if terminated or turnicated:
            return reward == 1

    return False
    pass

# ── Step 016  evaluate_success_rate ──
def evaluate_success_rate(env, policy, num_episodes, seed=0, max_steps=200):
    # TODO: run num_episodes greedy rollouts and return the fraction that reached the goal.
    success = 0

    for i in range(num_episodes):
        if run_greedy_episode(env, policy, seed + i, max_steps):
            success += 1

    return success / num_episodes
    pass

# ── Scaffold (runner) ──
"""Q-Learning on FrozenLake: train a tabular agent and evaluate its greedy policy."""
import numpy as np
import gymnasium as gym

from solution import (
    init_q_table,
    max_q_value,
    greedy_action,
    sample_random_action,
    should_explore,
    epsilon_greedy_action,
    decay_epsilon,
    td_target,
    td_error,
    q_learning_update,
    interaction_step,
    run_training_episode,
    train_q_learning,
    extract_greedy_policy,
    run_greedy_episode,
    evaluate_success_rate,
)


if __name__ == "__main__":
    np.random.seed(0)

    # Build a non-slippery FrozenLake for faster, more reliable learning.
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.action_space.seed(0)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(f"FrozenLake: {num_states} states, {num_actions} actions")

    # Train the tabular Q-learning agent.
    q_table, reward_history = train_q_learning(
        env,
        num_episodes=2000,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=0,
        max_steps=200,
    )
    print(f"Q-table shape: {q_table.shape}")
    print(f"Reward history length: {len(reward_history)}")
    early_avg = float(np.mean(reward_history[:100]))
    late_avg = float(np.mean(reward_history[-100:]))
    print(f"Mean reward first 100 episodes: {early_avg:.3f}")
    print(f"Mean reward last 100 episodes:  {late_avg:.3f}")

    # Extract greedy policy and inspect a couple of Q-values.
    policy = extract_greedy_policy(q_table)
    print(f"Greedy policy (first 8 states): {policy[:8].tolist()}")
    print(f"Greedy action at state 0: {greedy_action(q_table, 0)}")
    print(f"Max Q-value at state 0: {max_q_value(q_table, 0):.4f}")

    # Run one greedy episode and report success.
    reached_goal = run_greedy_episode(env, policy, seed=0, max_steps=200)
    print(f"Single greedy episode reached goal: {bool(reached_goal)}")

    # Evaluate success rate over many greedy episodes.
    success_rate = evaluate_success_rate(env, policy, num_episodes=100, seed=0, max_steps=200)
    print(f"Greedy success rate over 100 episodes: {success_rate:.2f}")

    env.close()
