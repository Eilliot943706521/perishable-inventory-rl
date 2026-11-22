"""Training reward scaling must preserve the physical environment."""
from dataclasses import replace
import numpy as np
from env.inventory_env import InventoryConfig
from env.seasonal_price import SeasonalPriceSchedule
from experiments.seasonal_ppo_common import RewardScale


def test_reward_scale_preserves_state_rng_costs_and_timeout():
    c=replace(InventoryConfig.from_yaml('configs/base_config.yaml'),episode_steps=6)
    s=SeasonalPriceSchedule([2,1],[.8,1.2],[.8,1.2])
    plain=s.make_env(c);wrapped=RewardScale(s.make_env(c),.01)
    for e in (plain,wrapped):e.reset(seed=42)
    for _ in range(6):
        a=plain.step([2,1,1,2]);b=wrapped.step([2,1,1,2])
        np.testing.assert_array_equal(a[0],b[0])
        assert a[2:4]==b[2:4]
        assert b[1]==a[1]*.01
        assert a[4]['costs']==b[4]['costs']
        assert a[4]['total_demand']==b[4]['total_demand']
    plain.close();wrapped.close()
