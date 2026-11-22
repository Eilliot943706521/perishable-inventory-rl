"""Three scenarios with fixed PPO tuning, matching original training protocol.

Training reward scaling only; Monitor and evaluations report original cash.
Scenarios all have four observations (three stocks plus calendar day).
"""
import argparse,json,shutil,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch,yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.inventory_env import InventoryConfig
from env.seasonal_price import SeasonalPriceSchedule
from experiments.seasonal_ppo_common import RewardScale,evaluate,protocol,INITIALS,TEST_SEEDS

SCENARIOS={'price_only':'seasonal_price_only_09_11.yaml',
           'demand_only':'seasonal_demand_only_09_11.yaml',
           'demand_and_price':'seasonal_demand_price_09_11.yaml'}
SEEDS=list(range(142,147))


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def job(args):
    root=Path(args.run_dir);output=root/args.scenario/f'seed_{args.seed}'
    output.mkdir(parents=True,exist_ok=False)
    c=InventoryConfig.from_yaml(root/'configs/base_config.yaml')
    schedule=SeasonalPriceSchedule.from_yaml(root/'configs'/SCENARIOS[args.scenario])
    params=dict(yaml.safe_load((root/'configs/ppo_seasonal_scaled_entropy.yaml').read_text()))
    scale=params.pop('training_reward_scale');policy=params.pop('policy');arch=params.pop('net_arch');steps=params.pop('total_timesteps')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    write(output/'status.json',{'state':'training','seed':args.seed,'scenario':args.scenario})
    environment=Monitor(schedule.make_env(c),filename=str(output/'monitor'))
    env=RewardScale(environment,scale)
    model=PPO(policy,env,seed=args.seed,device='cpu',verbose=1,policy_kwargs={'net_arch':arch},**params)
    started=time.perf_counter();model.learn(total_timesteps=steps);seconds=time.perf_counter()-started
    assert model.num_timesteps==steps
    metrics={k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and isinstance(v,(int,float,np.number))}
    model.save(output/'model');env.close()
    training={'seed':args.seed,'scenario':args.scenario,'environment':asdict(c),'calendar':schedule.to_dict(),'requested_steps':steps,'actual_steps':model.num_timesteps,'training_reward_scale':scale,'ppo':params,'policy':policy,'net_arch':arch,'training_seconds':seconds,'device':'cpu','final_training_metrics':metrics,'evaluation_units':'original cash profit'}
    write(output/'training.json',training);write(output/'status.json',{'state':'evaluating','seed':args.seed,'scenario':args.scenario})
    result={'training':training,'protocol':protocol(c,schedule,TEST_SEEDS),'evaluation':{}}
    for mode in ['deterministic','stochastic']:
        result['evaluation'][mode]={}
        for index,(name,initial) in enumerate(INITIALS.items()):
            torch.manual_seed(100_000+10*args.seed+index)
            row=evaluate(c,schedule,lambda obs:model.predict(obs,deterministic=mode=='deterministic')[0],initial)
            result['evaluation'][mode][name]=row
            print(args.scenario,args.seed,mode,name,row['annual_mean'],flush=True)
    write(output/'result.json',result);write(output/'status.json',{'state':'complete','seed':args.seed,'scenario':args.scenario})


def summarize(root):
    summaries={}
    for scenario in SCENARIOS:
        rows=[json.loads((root/scenario/f'seed_{seed}/result.json').read_text()) for seed in SEEDS]
        assert all(row['protocol']==rows[0]['protocol'] for row in rows)
        summary={'scenario':scenario,'training_seeds':SEEDS,'protocol':rows[0]['protocol'],'evaluation':{},'EV_by_seed':[r['training']['final_training_metrics'].get('train/explained_variance') for r in rows]}
        for mode in ['deterministic','stochastic']:
            summary['evaluation'][mode]={}
            for name in INITIALS:
                profits=np.array([row['evaluation'][mode][name]['profits'] for row in rows])
                means=profits.mean(axis=1)
                summary['evaluation'][mode][name]={'annual_mean':float(profits.mean()),'seed_means':means.tolist(),'training_seed_sd':float(means.std(ddof=1)),'profits_by_training_seed':profits.tolist()}
        write(root/scenario/'summary.json',summary);summaries[scenario]=summary
    rankings={}
    for mode in ['deterministic','stochastic']:
        rankings[mode]={}
        for initial in INITIALS:
            rankings[mode][initial]=sorted([{'scenario':name,'annual_mean':row['evaluation'][mode][initial]['annual_mean']} for name,row in summaries.items()],key=lambda x:x['annual_mean'],reverse=True)
    write(root/'comparison.json',{'primary_metric':'deterministic empty-start annual cash profit','rankings':rankings,'scenarios':summaries,'note':'Different economic environments; ranking is descriptive profit, not proof that one algorithm is superior. Same demand seeds, initial conditions, budget and hyperparameters.'})
    return rankings


def run(args):
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();(root/'configs').mkdir()
    for name in ['base_config.yaml','ppo_seasonal_scaled_entropy.yaml',*SCENARIOS.values()]:shutil.copyfile(Path('configs')/name,root/'configs'/name)
    status={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'workers':args.workers,'seeds':SEEDS,'steps_per_seed':73000,'training_reward_scale':.01,'ent_coef':.01,'active':{},'finished':[]};lock=threading.Lock();started=time.perf_counter()
    def save():
        status['elapsed_seconds']=time.perf_counter()-started;write(root/'pipeline_status.json',status)
    def worker(item):
        scenario,seed=item;key=f'{scenario}_{seed}';logfile=root/'logs'/f'{key}.log'
        with lock:status['active'][key]={'scenario':scenario,'seed':seed,'log':str(logfile)};save()
        with logfile.open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-B','-m','experiments.compare_seasonal_scenarios','job','--run-dir',str(root),'--scenario',scenario,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
            with lock:status['active'][key]['pid']=child.pid;save()
            code=child.wait()
        if code:raise RuntimeError(f'{key} failed: {logfile}')
        with lock:del status['active'][key];status['finished'].append(key);save()
        print('FINISHED',key,round(time.perf_counter()-started,1),flush=True)
    try:
        # Interleave scenarios, so all groups progress on matching seeds.
        items=[(scenario,seed) for seed in SEEDS for scenario in SCENARIOS]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(worker,items))
        rankings=summarize(root);status.update(state='complete',completed_utc=datetime.now(timezone.utc).isoformat());save()
        print('ALL COMPLETE',json.dumps(rankings),flush=True)
    except BaseException as exc:
        status.update(state='failed',error=str(exc));save();traceback.print_exc();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['run','job','summary'])
    p.add_argument('--run-dir',default='results/seasonal_price_extension/comparison_09_11_scaled_20261122')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--scenario',choices=list(SCENARIOS));p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.task=='job' and (args.scenario is None or args.seed not in SEEDS):p.error('job requires a scenario and seed 142..146')
    if not 1<=args.workers<=4:p.error('use 1..4 workers')
    if args.task=='job':job(args)
    elif args.task=='run':run(args)
    else:summarize(Path(args.run_dir))

if __name__=='__main__':main()
