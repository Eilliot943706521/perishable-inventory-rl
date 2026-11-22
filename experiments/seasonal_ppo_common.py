"""Shared evaluation and reward wrapper for the three retained seasonal runs.

Functions were moved verbatim from the completed run's helpers. RewardScale
changes training reward only; evaluate reports unscaled cash from a fresh env.
"""
from dataclasses import asdict
import json

import gymnasium as gym
import numpy as np

INITIALS = {"empty": [0, 0, 0], "nonempty": [3, 2.3, 1.7]}
TEST_SEEDS = list(range(42, 142))


class RewardScale(gym.RewardWrapper):
    def __init__(self,env,scale):
        super().__init__(env);self.scale=scale
    def reward(self,reward):return float(reward)*self.scale


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def protocol(config, schedule, seeds):
    return canonical({"environment": asdict(config), "calendar": schedule.to_dict(),
                      "test_seeds": list(seeds), "initials": INITIALS,
                      "metric": "undiscounted cash profit over episode_steps; starts on cycle day 0"})


def evaluate(config, schedule, actions, initial, seeds=TEST_SEEDS):
    """Same demand seeds and initial calendar phase for every method."""
    if config.shelf_life != 3 or config.lead_time != 1:
        raise ValueError("seasonal comparison supports M=3,L=1 only")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    envs = [schedule.make_env(config) for _ in seeds]
    try:
        states = np.array([env.reset(seed=seed, options={"initial_state": initial})[0]
                           for env, seed in zip(envs, seeds)])
        profits = np.zeros(len(envs))
        # season x [days, profit, order, sales, lost sales, expired]
        ledger = np.zeros((len(schedule.season_lengths), 6))
        price_counts = np.zeros((3, len(config.discount_percent)), dtype=int)
        for step in range(config.episode_steps):
            batch = np.asarray(actions(states))
            if batch.shape != (len(envs), 4) or not np.issubdtype(batch.dtype, np.integer):
                raise ValueError("policy must return integer [order, price1, price2, price3] actions")
            if any(not env.action_space.contains(action) for env, action in zip(envs, batch)):
                raise ValueError("policy returned an action outside the environment action space")
            rows, ages = np.nonzero(states[:, :3] > 0)
            np.add.at(price_counts, (ages, batch[rows, ages + 1]), 1)
            for i, (env, action) in enumerate(zip(envs, batch)):
                states[i], reward, terminated, truncated, info = env.step(action)
                if terminated or truncated != (step == config.episode_steps - 1):
                    raise RuntimeError("unexpected episode boundary")
                if not np.isclose(reward, info["revenue"] - sum(info["costs"].values())):
                    raise ArithmeticError("profit ledger mismatch")
                profits[i] += reward
                ledger[info["season_index"]] += [
                    1, reward, info["order"], np.sum(info["sales_by_age"]),
                    info["lost_sales"], info["expired"]]
        return {"profits": profits.tolist(), "annual_mean": float(profits.mean()),
                "available_price_counts": price_counts.tolist(),
                "season_totals_per_episode": (ledger / len(envs)).tolist(),
                "season_ledger_columns": ["days", "profit", "orders", "sales", "lost_sales", "expired"]}
    finally:
        for env in envs:
            env.close()
