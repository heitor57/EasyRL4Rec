import os
import sys
import pickle
import pandas as pd
import pathlib
import numpy as np

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from src.core.envs.BaseData import BaseData, get_distance_mat

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/SiTunes"
DATAPATH = pathlib.Path(os.path.join(ROOTPATH, "raw"))
PRODATAPATH = pathlib.Path(os.path.join(ROOTPATH, "processed"))

for path in [PRODATAPATH]:
    if not os.path.exists(path):
        os.mkdir(path)

class SiTunesData(BaseData):
    def __init__(self):
        super(SiTunesData, self).__init__()
        self.train_data_path = "train.parquet"
        self.val_data_path = "test.parquet"
        self.num_users = 30
    
    def get_features(self, is_userinfo=True):
        user_features = ["user_id", 'gender_u', 'age', 'location', 'fashioninterest']
        if not is_userinfo:
            user_features = ["user_id"]
        item_features = ['item_id', 'gender_i', "jackettype", 'color', 'onfrontpage']
        reward_features = ["rating"]
        return user_features, item_features, reward_features
        
    def get_df(self, name="train.parquet"):
        # read interaction
        filename = os.path.join(PRODATAPATH, name)
        df_data = pd.read_parquet(filename)
        # df_data = pd.DataFrame([], columns=["user_id", "item_id", "rating"])

        # for item in mat_train.columns:
        #     one_item = mat_train.loc[mat_train[item] > 0, item].reset_index().rename(
        #         columns={"index": "user_id", item: "rating"})
        #     one_item["item_id"] = item
        #     df_data = pd.concat([df_data, one_item])
        # df_data.reset_index(drop=True, inplace=True)

        # read user feature
        df_user = self.load_user_feat()

        # read item features
        df_item = SiTunesData.load_item_feat()

        # df_data = df_data.join(df_user, on="user_id", how='left')
        # df_data = df_data.join(df_item, on="item_id", how='left')

        # df_data = df_data.astype(int)
        list_feat = None

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        df_data, _, df_item, _ = self.get_df("train.pickle")
        feature_domination_path = os.path.join(PRODATAPATH, "feature_domination.pickle")

        if os.path.isfile(feature_domination_path):
            item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
        else:
            item_feat_domination = self.get_sorted_domination_features(
                df_data, df_item, is_multi_hot=False)
            pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))
        return item_feat_domination
    
    def get_item_similarity(self):
        item_similarity_path = os.path.join(PRODATAPATH, "item_similarity.pickle")
        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            mat = SiTunesData.load_mat()
            mat_distance = SiTunesData.get_saved_distance_mat(mat, PRODATAPATH)
            item_similarity = 1 / (mat_distance + 1)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
      
    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            df_data, df_user, df_item, list_feat = self.get_df("train.ascii")

            n_users = df_data['user_id'].nunique()
            n_items = df_data['item_id'].nunique()

            df_data_filtered = df_data[df_data["rating"]>=3.]
            
            groupby = df_data_filtered.loc[:, ["user_id", "item_id"]].groupby(by="item_id")
            df_pop = groupby.user_id.apply(list).reset_index()
            df_pop["popularity"] = df_pop['user_id'].apply(lambda x: len(x) / n_users)

            item_pop_df = pd.DataFrame(np.arange(n_items), columns=["item_id"])
            item_pop_df = item_pop_df.merge(df_pop, how="left", on="item_id")
            item_pop_df['popularity'].fillna(0, inplace=True)
            item_popularity = item_pop_df['popularity']
            pickle.dump(item_popularity, open(item_popularity_path, 'wb'))
        
        return item_popularity

    def load_user_feat(self):
        df_user = pd.DataFrame(np.arange(self.num_users), columns=["user_id"])
        df_user.set_index("user_id", inplace=True)
        return df_user

    @staticmethod
    def load_item_feat():
        df_item = pd.read_csv(DATAPATH / "music_metadata"/ "music_info.csv")
        return df_item
    
    @staticmethod
    def load_mat():
        df_user = pd.DataFrame(np.arange(num_users), columns=["user_id"])
        df_user.set_index("user_id", inplace=True)
        return df_user


 
if __name__ == "__main__":
    dataset = SiTunesData()
    df_train, df_user_train, df_item_train, _ = dataset.get_train_data()
    df_val, df_user_val, df_item_val, _ = dataset.get_val_data()
    print("SiTunes: Train #user={}  #item={}  #inter={}".format(df_train['user_id'].nunique(), df_train['item_id'].nunique(), len(df_train)))
    print("SiTunes: Test  #user={}  #item={}  #inter={}".format(df_val['user_id'].nunique(), df_val['item_id'].nunique(), len(df_val)))
    print(df_user_train)
    print(df_item_train)
    print(df_train)
    print('=====================dataset.get_features()')
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
