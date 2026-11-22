"""Demand+price seasonality equivalence, cash-ledger and boundary checks."""
from dataclasses import replace
import numpy as np
import pytest

from env.inventory_env import InventoryConfig, InventoryEnv
from env.seasonal_extension import SeasonSchedule
from env.seasonal_price import SeasonalPriceSchedule


def config(**changes):
    return replace(InventoryConfig.from_yaml('configs/base_config.yaml'), **changes)


def test_price_unit_multipliers_exactly_match_previous_seasonal_environment():
    c=config(episode_steps=6)
    old=SeasonSchedule([2,1],[.8,1.2]).make_env(c)
    new=SeasonalPriceSchedule([2,1],[.8,1.2],[1,1]).make_env(c)
    for env in (old,new): env.reset(seed=42,options={'initial_state':[3,2.3,1.7]})
    for _ in range(6):
        a=old.step([2,0,1,2]);b=new.step([2,0,1,2])
        np.testing.assert_array_equal(a[0],b[0])
        assert a[1:4]==b[1:4]
        for key in ['total_demand','sales_by_age','demand_shares','prices']:
            np.testing.assert_array_equal(a[4][key],b[4][key])
    old.close();new.close()


def test_actual_prices_drive_both_mnl_and_cash_revenue_with_fixed_reference():
    c=config(episode_steps=1)
    env=SeasonalPriceSchedule([1],[1.2],[1.2]).make_env(c)
    baseline=InventoryEnv(c)
    stock=np.array([3.,2.3,1.7])
    for e in (env,baseline):e.reset(seed=17,options={'initial_state':stock})
    obs,reward,_,_,info=env.step([2,0,1,2])
    _,_,_,_,control=baseline.step([2,0,1,2])
    prices=np.array([12.,8.4,6.])
    weights=np.exp(2-np.array([3,5,7])*prices/10)
    shares=weights/(1+weights.sum())
    potential=control['total_demand']*1.2
    demand=potential*shares
    sales=np.minimum(stock,demand)
    remaining=stock-sales
    expected=np.dot(prices,sales)-3*2-remaining[:2].sum()-np.maximum(demand-stock,0).sum()-remaining[2]
    np.testing.assert_allclose(info['prices'],prices)
    np.testing.assert_allclose(info['demand_prices'],[1.2,.84,.6])
    np.testing.assert_allclose(info['demand_shares'],shares)
    assert info['demand_price_reference']==10
    assert info['total_demand']==pytest.approx(potential)
    assert reward==pytest.approx(expected)
    np.testing.assert_allclose(obs[:3],[2,remaining[0],remaining[1]])
    assert not np.allclose(info['demand_shares'],control['demand_shares'])
    env.close();baseline.close()


def test_price_and_demand_switch_on_same_day_and_wrap_without_changing_config():
    c=config(episode_steps=6)
    env=SeasonalPriceSchedule([2,1],[.8,1.2],[.9,1.1]).make_env(c)
    raw=InventoryEnv(c)
    env.reset(seed=11);raw.reset(seed=11)
    for day in range(6):
        obs,_,_,_,info=env.step([2,0,1,2]);base_info=raw.step([2,0,1,2])[4]
        season=0 if day%3<2 else 1
        np.testing.assert_allclose(info['prices'],np.array([10,7,5])*[.9,1.1][season])
        assert info['total_demand']==pytest.approx(base_info['total_demand']*[.8,1.2][season])
        assert info['season_index']==season and info['cycle_day']==day%3
        assert info['price_multiplier']==[.9,1.1][season]
        assert obs[3]==((day+1)%3)/2
        assert env.unwrapped.config.regular_price==10
    env.close();raw.close()


@pytest.mark.parametrize('prices',[[0],[-1],[np.nan],[np.inf],[True],[],[1,2]])
def test_invalid_price_schedules(prices):
    with pytest.raises(ValueError):SeasonalPriceSchedule([1],[1],prices)


def test_price_schedule_roundtrip_and_missing_price_rejected():
    s=SeasonalPriceSchedule.from_yaml('configs/seasonal_demand_price_09_11.yaml')
    assert SeasonalPriceSchedule.from_dict(s.to_dict())==s
    with pytest.raises(ValueError):SeasonalPriceSchedule.from_dict({'season_lengths':[1],'demand_multipliers':[1]})


def test_reset_restores_first_season_and_seed():
    env=SeasonalPriceSchedule([1,1],[.8,1.2],[.8,1.2]).make_env(config(episode_steps=2))
    outcomes=[]
    for _ in range(2):
        obs,_=env.reset(seed=3)
        assert obs[3]==0
        outcomes.append(env.step([2,0,1,2]))
    np.testing.assert_array_equal(outcomes[0][0],outcomes[1][0])
    assert outcomes[0][1]==outcomes[1][1]
    assert outcomes[0][4]['price_multiplier']==outcomes[1][4]['price_multiplier']==.8
    env.close()
