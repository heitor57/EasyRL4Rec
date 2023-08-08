import argparse
import functools
import os
import pprint
import sys
import traceback

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, prepare_dir_log, prepare_user_model, prepare_envs, setup_state_tracker

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from core.collector.collector_set import CollectorSet
from core.evaluation.evaluator import Callback_Coverage_Count
from core.util.data import get_common_args, get_val_data, get_training_item_domination
from core.collector.collector import Collector
from core.policy.a2c import A2CPolicy_withEmbedding
from core.trainer.onpolicy import onpolicy_trainer

from tianshou.data import VectorReplayBuffer

from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

# from util.upload import my_upload
from util.utils import LoggerCallback_Policy, save_model_fn
import logzero
from logzero import logger

try:
    import envpool
except ImportError:
    envpool = None


def get_args_ips_policy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_model_name", type=str, default="DeepFM-IPS")
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument("--model_name", type=str, default="IPS")
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    
    parser.add_argument("--entropy_window", type=int, nargs="*", default=[])
    parser.add_argument('--lambda_variance', default=0, type=float)
    parser.add_argument('--lambda_entropy', default=0, type=float)

    parser.add_argument("--read_message", type=str, default="DeepFM-IPS")
    parser.add_argument("--message", type=str, default="IPS")
    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    optim_RL = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    dist = torch.distributions.Categorical
    policy = A2CPolicy_withEmbedding(
        actor,
        critic,
        optim,
        dist,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=args.action_shape,
        action_bound_method="",  # not clip
        action_scaling=False
    )
    policy.set_eps(args.eps)

    # Prepare the collectors and logs
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
    )
    # test_collector = Collector(
    #     policy, test_envs_dict,
    #     VectorReplayBuffer(args.buffer_size, len(test_envs)),
    #     preprocess_fn=state_tracker.build_state,
    #     exploration_noise=args.exploration_noise,
    # )
    policy.set_collector(train_collector)

    test_collector_set = CollectorSet(policy, test_envs_dict, args.buffer_size, args.test_num,
                                      preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return policy, train_collector, test_collector_set, optim


def learn_policy(args, env, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path):
    # log
    # log_path = os.path.join(args.logdir, args.env, 'a2c')
    # writer = SummaryWriter(log_path)
    # logger1 = TensorboardLogger(writer)
    # def save_best_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # env = test_collector_set.env
    df_val, df_user_val, df_item_val, list_feat = get_val_data(args.env)
    item_feat_domination = get_training_item_domination(args.env)
    policy.callbacks = [
        Callback_Coverage_Count(test_collector_set, df_item_val, args.need_transform, item_feat_domination,
                                lbe_item=env.lbe_item if args.need_transform else None, top_rate=args.top_rate, draw_bar=args.draw_bar),
        LoggerCallback_Policy(logger_path, args.force_length)]
    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector_set,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        # stop_fn=stop_fn,
        # save_best_fn=save_best_fn,
        # logger=logger1,
        save_model_fn=functools.partial(save_model_fn,
                                        model_save_path=model_save_path,
                                        state_tracker=state_tracker,
                                        optim=optim,
                                        is_save=args.is_save)
    )

    print(__file__)
    pprint.pprint(result)
    logger.info(result)


def main(args):
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, train_envs, test_envs_dict = prepare_envs(args, ensemble_models)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path)


if __name__ == '__main__':
    args_all = get_args_all()
    args = get_common_args(args_all)
    args_IPS = get_args_ips_policy()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_IPS.__dict__)

    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
