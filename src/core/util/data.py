import argparse

def get_env_args(args):
    env = args.env

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')

    parser.add_argument('--is_binarize', dest='is_binarize', action='store_true')
    parser.add_argument('--no_binarize', dest='is_binarize', action='store_false')

    parser.add_argument('--is_need_transform', dest='need_transform', action='store_true')
    parser.add_argument('--no_need_transform', dest='need_transform', action='store_false')

    parser.add_argument('--is_use_auxiliary', dest='use_auxiliary', action='store_true')
    parser.add_argument('--no_use_auxiliary', dest='use_auxiliary', action='store_false')
    parser.set_defaults(use_auxiliary=False)

    if env == "CoatEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=6, type=float)
        parser.add_argument('--num_leave_compute', default=7, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "YahooEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=120, type=float)
        parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "MovieLensEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=True)
        parser.set_defaults(use_auxiliary=True)
        # args.entropy_on_user = True
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=75, type=float)
        parser.add_argument('--num_leave_compute', default=7, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "KuaiRand-v0":
        parser.set_defaults(is_userinfo=False)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=False)
        # args.entropy_on_user = False
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--rating_threshold", type=float, default=1)
        parser.add_argument("--yfeat", type=str, default="is_click")

        parser.add_argument('--leave_threshold', default=0, type=float)
        parser.add_argument('--num_leave_compute', default=10, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)

    elif env == "KuaiEnv-v0":
        parser.set_defaults(is_userinfo=False)
        parser.set_defaults(is_binarize=False)
        parser.set_defaults(need_transform=True)
        # args.entropy_on_user = False
        parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--yfeat", type=str, default="watch_ratio_normed")

        # parser.add_argument('--leave_threshold', default=1, type=float)
        # parser.add_argument('--num_leave_compute', default=3, type=int)
        parser.add_argument('--leave_threshold', default=1, type=float)
        parser.add_argument('--num_leave_compute', default=9, type=int)
        parser.add_argument('--max_turn', default=30, type=int)
        # parser.add_argument('--window_size', default=3, type=int)
    elif env == "SiTunesEnv-v0":
        parser.set_defaults(is_userinfo=True)
        parser.set_defaults(is_binarize=True)
        parser.set_defaults(need_transform=True)
        parser.set_defaults(use_auxiliary=True)
        # args.entropy_on_user = True
        # parser.add_argument("--entropy_window", type=int, nargs="*", default=[1,2])
        parser.add_argument("--rating_threshold", type=float, default=4)
        parser.add_argument("--yfeat", type=str, default="rating")

        parser.add_argument('--leave_threshold', default=75, type=float)
        parser.add_argument('--num_leave_compute', default=7, type=int)
        parser.add_argument('--max_turn', default=30, type=int)

    parser.add_argument('--force_length', type=int, default=10)
    parser.add_argument("--top_rate", type=float, default=0.8)

    args_new = parser.parse_known_args()[0]
    args.__dict__.update(args_new.__dict__)

    return args


def get_true_env(args, read_user_num=None):
    if args.env == "CoatEnv-v0":
        from src.core.envs.Coat.CoatEnv import CoatEnv
        from src.core.envs.Coat.CoatData import CoatData
        mat, df_item, mat_distance = CoatEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "df_item": df_item,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}
        env = CoatEnv(**kwargs_um)
        dataset = CoatData()
    elif args.env == "YahooEnv-v0":
        from src.core.envs.YahooR3.YahooEnv import YahooEnv
        from src.core.envs.YahooR3.YahooData import YahooData
        mat, mat_distance = YahooEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}

        env = YahooEnv(**kwargs_um)
        dataset = YahooData()
    elif args.env == "MovieLensEnv-v0":
        from src.core.envs.MovieLens.MovieLensEnv import MovieLensEnv
        from src.core.envs.MovieLens.MovieLensData import MovieLensData
        mat, lbe_user, lbe_item, mat_distance = MovieLensEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "lbe_user": lbe_user,
                     "lbe_item": lbe_item,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}

        env = MovieLensEnv(**kwargs_um)
        dataset = MovieLensData()
    elif args.env == "KuaiRand-v0":
        from src.core.envs.KuaiRand_Pure.KuaiRandEnv import KuaiRandEnv
        from src.core.envs.KuaiRand_Pure.KuaiRandData import KuaiRandData
        mat, list_feat, mat_distance = KuaiRandEnv.load_env_data(args.yfeat, read_user_num=read_user_num)
        kwargs_um = {"yname": args.yfeat,
                     "mat": mat,
                     "mat_distance": mat_distance,
                     "list_feat": list_feat,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}
        env = KuaiRandEnv(**kwargs_um)
        dataset = KuaiRandData()
    elif args.env == "KuaiEnv-v0":
        from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv
        from src.core.envs.KuaiRec.KuaiData import KuaiData
        mat, lbe_user, lbe_item, list_feat, df_dist_small = KuaiEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "lbe_user": lbe_user,
                     "lbe_item": lbe_item,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init,
                     "list_feat": list_feat,
                     "df_dist_small": df_dist_small}
        env = KuaiEnv(**kwargs_um)
        dataset = KuaiData()
    elif args.env == "SiTunesEnv-v0":
        from src.core.envs.SiTunes.SiTunesEnv import SiTunesEnv
        from src.core.envs.SiTunes.SiTunesData import SiTunesData
        # mat, lbe_user, lbe_item, list_feat, df_dist_small = SiTunesEnv.load_env_data()
        # kwargs_um = {"mat": mat,
        #              "lbe_user": lbe_user,
        #              "lbe_item": lbe_item,
        #              "num_leave_compute": args.num_leave_compute,
        #              "leave_threshold": args.leave_threshold,
        #              "max_turn": args.max_turn,
        #              "random_init": args.random_init,
        #              "list_feat": list_feat,
        #              "df_dist_small": df_dist_small}
        # env = SiTunesEnv(**kwargs_um)
        mat, df_item, mat_distance = SiTunesEnv.load_env_data()
        kwargs_um = {"mat": mat,
                     "df_item": df_item,
                     "mat_distance": mat_distance,
                     "num_leave_compute": args.num_leave_compute,
                     "leave_threshold": args.leave_threshold,
                     "max_turn": args.max_turn,
                     "random_init": args.random_init}
        env = SiTunesEnv(**kwargs_um)
        dataset = SiTunesData()
    return env, dataset, kwargs_um