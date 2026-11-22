"""Shared schedule handling for the separate seasonal experiment entry points.

Only potential demand changes. Prices, costs and demand CV keep their baseline
meaning. The original env/seasonal.py is reused without modification.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .seasonal import SeasonalInventoryEnv


@dataclass(frozen=True)
class SeasonSchedule:
    season_lengths: tuple
    demand_multipliers: tuple

    def __post_init__(self):
        lengths, multipliers = tuple(self.season_lengths), tuple(self.demand_multipliers)
        if (not lengths or len(lengths) != len(multipliers)
                or any(type(n) is not int or n < 1 for n in lengths)):
            raise ValueError("use matching nonempty lists of positive integer season lengths")
        if any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in multipliers):
            raise ValueError("demand multipliers must be positive finite numbers")
        object.__setattr__(self, "season_lengths", lengths)
        object.__setattr__(self, "demand_multipliers", tuple(float(x) for x in multipliers))

    @classmethod
    def from_yaml(cls, path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"season_lengths", "demand_multipliers"}:
            raise ValueError("schedule requires only season_lengths and demand_multipliers")
        return cls(tuple(data["season_lengths"]), tuple(data["demand_multipliers"]))

    def to_dict(self):
        return {"season_lengths": list(self.season_lengths),
                "demand_multipliers": list(self.demand_multipliers)}

    @property
    def cycle_days(self):
        return sum(self.season_lengths)

    @property
    def daily_multipliers(self):
        return np.repeat(self.demand_multipliers, self.season_lengths)

    def make_env(self, config):
        return SeasonalInventoryEnv(config, list(self.season_lengths),
                                    list(self.demand_multipliers))
