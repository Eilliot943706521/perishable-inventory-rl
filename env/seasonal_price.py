"""Independent seasonal demand AND actual-price environment.

Price multipliers change both MNL demand utility and cash revenue. The demand
price reference remains the original config.demand_price_scale (10 in baseline).
Original stationary and demand-only environments are not modified.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .demand_model import demand_shares
from .seasonal import SeasonalInventoryEnv, _SeasonalDemandEnv
from .seasonal_extension import SeasonSchedule
from .state_transition import transition


class _SeasonalPriceDemandEnv(_SeasonalDemandEnv):
    def __init__(self, config, day_multipliers, day_price_multipliers):
        super().__init__(config, day_multipliers)
        self.day_price_multipliers = day_price_multipliers

    def step(self, action):
        if self._state is None or self._elapsed >= self.config.episode_steps:
            raise RuntimeError("reset is required before stepping")
        raw = np.asarray(action)
        if not np.issubdtype(raw.dtype, np.integer) or not self.action_space.contains(raw):
            raise ValueError("action requires valid integer order and discount indices")
        c = self.config
        day = self._elapsed % len(self.day_price_multipliers)
        price_multiplier = float(self.day_price_multipliers[day])
        base_prices = c.regular_price * (1.0 - np.asarray(c.discount_percent)[raw[1:]] / 100.0)
        prices = base_prices * price_multiplier
        available = self._state[:c.shelf_life] > 0 if c.exclude_empty_from_mnl else None
        shares = demand_shares(prices, c.price_sensitivity, c.demand_price_scale,
                              available, c.purchase_utility)
        total_demand = self._sample_total_demand()
        self._state, reward, info = transition(
            self._state, int(raw[0]), prices, total_demand * shares,
            c.shelf_life, c.lead_time, c.costs)
        self._elapsed += 1
        info.update(total_demand=total_demand, demand_shares=shares.copy(),
                    outside_demand=total_demand * (1.0 - shares.sum()),
                    demand_prices=prices / c.demand_price_scale,
                    demand_price_basis=c.demand_price_basis,
                    demand_price_reference=c.demand_price_scale,
                    price_multiplier=price_multiplier, base_prices=base_prices.copy(),
                    cycle_day=day)
        return self._state.copy(), float(reward), False, self._elapsed >= c.episode_steps, info


class SeasonalPriceInventoryEnv(SeasonalInventoryEnv):
    def __init__(self, config, season_lengths, demand_multipliers, price_multipliers):
        prices = tuple(price_multipliers)
        if len(prices) != len(season_lengths) or any(
                isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x) or x <= 0
                for x in prices):
            raise ValueError("one positive finite price multiplier is required per season")
        super().__init__(config, season_lengths, demand_multipliers)
        self.price_multipliers = tuple(float(x) for x in prices)
        self.env.close()
        self.env = _SeasonalPriceDemandEnv(
            config, np.repeat(self.multipliers, self.seasons),
            np.repeat(self.price_multipliers, self.seasons))

    @classmethod
    def from_yaml(cls, config, path):
        return SeasonalPriceSchedule.from_yaml(path).make_env(config)

    def save_checkpoint(self, path):
        raise NotImplementedError("use PPO model checkpoints; seasonal environment snapshots are unsupported")


@dataclass(frozen=True)
class SeasonalPriceSchedule(SeasonSchedule):
    price_multipliers: tuple

    def __post_init__(self):
        super().__post_init__()
        values = tuple(self.price_multipliers)
        if len(values) != len(self.season_lengths) or any(
                isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x) or x <= 0
                for x in values):
            raise ValueError("one positive finite price multiplier is required per season")
        object.__setattr__(self, "price_multipliers", tuple(float(x) for x in values))

    @classmethod
    def from_dict(cls, data):
        required = {"season_lengths", "demand_multipliers", "price_multipliers"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("schedule needs season_lengths, demand_multipliers and price_multipliers")
        return cls(tuple(data["season_lengths"]), tuple(data["demand_multipliers"]),
                   tuple(data["price_multipliers"]))

    def to_dict(self):
        return {**super().to_dict(), "price_multipliers": list(self.price_multipliers)}

    def make_env(self, config):
        return SeasonalPriceInventoryEnv(config, self.season_lengths,
                                         self.demand_multipliers, self.price_multipliers)
