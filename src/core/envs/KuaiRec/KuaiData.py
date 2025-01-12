import os
import sys
import pickle
from numba import njit
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix


sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from src.core.envs.BaseData import BaseData
from src.core.envs.KuaiRec.preprocessing_kuairec import get_df_data

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/KuaiRec"
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")

for path in [PRODATAPATH]:
    if not os.path.exists(path):
        os.mkdir(path)


class KuaiData(BaseData):
    def __init__(self):
        super(KuaiData, self).__init__()
        self.train_data_path = "big_matrix_processed.csv"
        self.val_data_path = "small_matrix_processed.csv"
    
    def get_features(self, is_userinfo=None):
        user_features = ["user_id"]
        item_features = ["item_id"] + ["feat" + str(i) for i in range(4)] + ["duration_normed"]
        reward_features = ["watch_ratio_normed"]
        return user_features, item_features, reward_features

    def get_df(self, name="big_matrix_processed.csv"):
        filename = os.path.join(DATAPATH, name)
        df_data = get_df_data(filename,
                              usecols=['user_id', 'item_id', 'timestamp', 'watch_ratio_normed', 'duration_normed'])

        # df_data['duration_normed'] = df_data['duration_ms'] / 1000

        # load feature info
        list_feat, df_feat = KuaiData.load_category()

        if name == "big_matrix_processed.csv":
            only_small = False
        else:
            only_small = True
        df_user = self.load_user_feat(only_small)
        df_item = self.load_item_feat(only_small)

        df_data = df_data.join(df_feat, on=['item_id'], how="left")

        # if is_require_feature_domination:
        #     item_feat_domination = self.get_domination(df_data, df_item)
        #     return df_data, df_user, df_item, list_feat, item_feat_domination

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        df_data, _, df_item, _ = self.get_df("big_matrix_processed.csv")
        feature_domination_path = os.path.join(PRODATAPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = self.get_sorted_domination_features(
                df_data, df_item, is_multi_hot=True, yname="watch_ratio_normed", threshold=0.6)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination
    
    def get_item_similarity(self):
        item_similarity_path = os.path.join(PRODATAPATH, "item_similarity.pickle")

        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            list_feat, df_feat = KuaiData.load_category()
            item_similarity = KuaiData.get_similarity_mat(list_feat)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
        
    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            filename = os.path.join(DATAPATH, "big_matrix_processed.csv")
            df_data = get_df_data(filename, usecols=['user_id', 'item_id', 'timestamp', 'watch_ratio'])
            n_users = df_data['user_id'].nunique()
            n_items = df_data['item_id'].nunique()
            
            df_data_filtered = df_data[df_data['watch_ratio']>=1.]
            
            groupby = df_data_filtered.loc[:, ["user_id", "item_id"]].groupby(by="item_id")
            df_pop = groupby.user_id.apply(list).reset_index()
            df_pop["popularity"] = df_pop['user_id'].apply(lambda x: len(x) / n_users)

            item_pop_df = pd.DataFrame(np.arange(n_items), columns=["item_id"])
            item_pop_df = item_pop_df.merge(df_pop, how="left", on="item_id")
            item_pop_df['popularity'].fillna(0, inplace=True)
            item_popularity = item_pop_df['popularity']
            pickle.dump(item_popularity, open(item_popularity_path, 'wb'))
        return item_popularity

    def load_user_feat(self, only_small=False):
        print("load user features")
        filepath = os.path.join(DATAPATH, 'user_features.csv')
        df_user = pd.read_csv(filepath, usecols=['user_id', 'user_active_degree',
                                                 'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                                                 'fans_user_num_range', 'friend_user_num_range',
                                                 'register_days_range'] + [f'onehot_feat{x}' for x in range(18)]
                              )
        for col in ['user_active_degree',
                    'is_live_streamer', 'is_video_author', 'follow_user_num_range',
                    'fans_user_num_range', 'friend_user_num_range', 'register_days_range']:

            df_user[col] = df_user[col].map(lambda x: chr(0) if x == 'UNKNOWN' else x)
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1
        for col in [f'onehot_feat{x}' for x in range(18)]:
            df_user.loc[df_user[col].isna(), col] = -124
            lbe = LabelEncoder()
            df_user[col] = lbe.fit_transform(df_user[col])
            # print(lbe.classes_)
            if chr(0) in lbe.classes_.tolist() or -124 in lbe.classes_.tolist():
                assert lbe.classes_[0] in {-124, chr(0)}
                # do not add one
            else:
                df_user[col] += 1

        df_user = df_user.set_index("user_id")

        if only_small:
            lbe_user, lbe_item = self.get_lbe()
            user_list = lbe_user.classes_
            df_user_env = df_user.loc[user_list]
            return df_user_env

        return df_user

    def load_item_feat(self, only_small=False):
        list_feat, df_feat = KuaiData.load_category()
        video_mean_duration = KuaiData.load_video_duration()
        df_item = df_feat.join(video_mean_duration, on=['item_id'], how="left")

        if only_small:
            lbe_user, lbe_item = self.get_lbe()
            item_list = lbe_item.classes_
            df_item_env = df_item.loc[item_list]
            return df_item_env

        return df_item

    def get_lbe(self):
        if not os.path.isfile(os.path.join(DATAPATH, "user_id_small.csv")) or not os.path.isfile(
                os.path.join(DATAPATH, "item_id_small.csv")):
            small_path = os.path.join(DATAPATH, "small_matrix_processed.csv")
            df_small = get_df_data(small_path, usecols=['user_id', 'item_id'])

            user_id_small = pd.DataFrame(df_small["user_id"].unique(), columns=["user_id_small"])
            item_id_small = pd.DataFrame(df_small["item_id"].unique(), columns=["item_id_small"])

            user_id_small.to_csv(os.path.join(DATAPATH, "user_id_small.csv"), index=False)
            item_id_small.to_csv(os.path.join(DATAPATH, "item_id_small.csv"), index=False)
        else:
            user_id_small = pd.read_csv(os.path.join(DATAPATH, "user_id_small.csv"))
            item_id_small = pd.read_csv(os.path.join(DATAPATH, "item_id_small.csv"))

        lbe_user = LabelEncoder()
        lbe_user.fit(user_id_small["user_id_small"])

        lbe_item = LabelEncoder()
        lbe_item.fit(item_id_small["item_id_small"])

        return lbe_user, lbe_item


    @staticmethod
    def load_mat():
        small_path = os.path.join(DATAPATH, "small_matrix_processed.csv")
        df_small = get_df_data(small_path, usecols=['user_id', 'item_id', 'watch_ratio'])
        # df_small['watch_ratio'][df_small['watch_ratio'] > 5] = 5
        df_small.loc[df_small['watch_ratio'] > 5, 'watch_ratio'] = 5

        lbe_item = LabelEncoder()
        lbe_item.fit(df_small['item_id'].unique())

        lbe_user = LabelEncoder()
        lbe_user.fit(df_small['user_id'].unique())

        mat = csr_matrix(
            (df_small['watch_ratio'],
             (lbe_user.transform(df_small['user_id']), lbe_item.transform(df_small['item_id']))),
            shape=(df_small['user_id'].nunique(), df_small['item_id'].nunique())).toarray()

        mat[np.isnan(mat)] = df_small['watch_ratio'].mean()
        mat[np.isinf(mat)] = df_small['watch_ratio'].mean()

        return mat, lbe_user, lbe_item

    @staticmethod
    def load_category(tag_label="tags"):
        # load categories:
        print("load item feature")
        filepath = os.path.join(DATAPATH, 'item_categories.csv')
        df_feat0 = pd.read_csv(filepath, header=0)
        df_feat0.feat = df_feat0.feat.map(eval)

        list_feat = df_feat0.feat.to_list()
        # df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3'], dtype=int)
        df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3'])
        df_feat.index.name = "item_id"
        df_feat[df_feat.isna()] = -1
        df_feat = df_feat + 1
        df_feat = df_feat.astype(int)
        df_feat[tag_label] = list_feat

        return list_feat, df_feat

    @staticmethod
    def load_video_duration():
        duration_path = os.path.join(DATAPATH, "video_duration_normed.csv")
        if os.path.isfile(duration_path):
            video_mean_duration = pd.read_csv(duration_path, header=0)["duration_normed"]
        else:
            small_path = os.path.join(DATAPATH, "small_matrix_processed.csv")
            small_duration = get_df_data(small_path, usecols=["item_id", 'duration_normed'])
            big_path = os.path.join(DATAPATH, "big_matrix_processed.csv")
            big_duration = get_df_data(big_path, usecols=["item_id", 'duration_normed'])
            duration_all = small_duration.append(big_duration)
            video_mean_duration = duration_all.groupby("item_id").agg(lambda x: sum(list(x)) / len(x))[
                "duration_normed"]
            video_mean_duration.to_csv(duration_path, index=False)

        video_mean_duration.index.name = "item_id"
        return video_mean_duration

    @staticmethod
    def get_similarity_mat(list_feat):
        similarity_mat_path = os.path.join(PRODATAPATH, "similarity_mat_video.csv")
        if os.path.isfile(similarity_mat_path):
            # with open(similarity_mat_path, 'rb') as f:
            #     similarity_mat = np.load(f, allow_pickle=True, fix_imports=True)
            print(f"loading similarity matrix... from {similarity_mat_path}")
            df_sim = pd.read_csv(similarity_mat_path, index_col=0)
            df_sim.columns = df_sim.columns.astype(int)
            print("loading completed.")
            similarity_mat = df_sim.to_numpy()
        else:
            series_feat_list = pd.Series(list_feat)
            df_feat_list = series_feat_list.to_frame("categories")
            df_feat_list.index.name = "video_id"

            similarity_mat = np.zeros([len(df_feat_list), len(df_feat_list)])
            print("Compute the similarity matrix (for the first time and will be saved for later usage)")
            for i in tqdm(range(len(df_feat_list)), desc="Computing..."):
                for j in range(i):
                    similarity_mat[i, j] = similarity_mat[j, i]
                for j in range(i, len(df_feat_list)):
                    similarity_mat[i, j] = len(set(series_feat_list[i]).intersection(set(series_feat_list[j]))) / len(
                        set(series_feat_list[i]).union(set(series_feat_list[j])))

            df_sim = pd.DataFrame(similarity_mat)
            df_sim.to_csv(similarity_mat_path)

        return similarity_mat

    @staticmethod
    def get_saved_distance_mat(list_feat, sub_index_list):
        if sub_index_list is not None:
            distance_mat_small_path = os.path.join(PRODATAPATH, "distance_mat_video_small.csv")
            if os.path.isfile(distance_mat_small_path):
                print("loading small distance matrix...")
                df_dist_small = pd.read_csv(distance_mat_small_path, index_col=0)
                df_dist_small.columns = df_dist_small.columns.astype(int)
                print("loading completed.")
            else:
                similarity_mat = KuaiData.get_similarity_mat(list_feat)
                df_sim = pd.DataFrame(similarity_mat)
                df_sim_small = df_sim.loc[sub_index_list, sub_index_list]

                df_dist_small = 1.0 / df_sim_small

                df_dist_small.to_csv(distance_mat_small_path)

            return df_dist_small

        return None





if __name__ == "__main__":
    dataset = KuaiData()
    df_train, df_user_train, df_item_train, _ = dataset.get_train_data()
    df_val, df_user_val, df_item_val, _ = dataset.get_val_data()
    print("KuaiRec: Train #user={}  #item={}  #inter={}".format(df_train['user_id'].nunique(), df_train['item_id'].nunique(), len(df_train)))
    print("KuaiRec: Test  #user={}  #item={}  #inter={}".format(df_val['user_id'].nunique(), df_val['item_id'].nunique(), len(df_val)))
    print(df_user_train)
    print(df_item_train)
    print(df_train)
    print('dataset.get_features()')
    print(dataset.get_features())
    print('dataset.get_df()')
    print(dataset.get_df())
    print('dataset.get_domination()')
    print(dataset.get_domination())
    print('dataset.get_item_similarity()')
    print(dataset.get_item_similarity())
    print('dataset.get_item_popularity()')
    print(dataset.get_item_popularity())
    print('dataset.load_user_feat()')
    print(dataset.load_user_feat())
    print('dataset.load_item_feat()')
    print(dataset.load_item_feat())
    print('dataset.load_mat()')
    print(dataset.load_mat())

