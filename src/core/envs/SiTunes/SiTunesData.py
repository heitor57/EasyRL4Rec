import os
import sys
import pickle
import pandas as pd
import pathlib
import numpy as np
from scipy.sparse import csr_matrix

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from sklearn.preprocessing import LabelEncoder
from src.core.envs.BaseData import BaseData, get_distance_mat

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/SiTunes"
DATAPATH = pathlib.Path(os.path.join(ROOTPATH, "raw"))
PRODATAPATH = pathlib.Path(os.path.join(ROOTPATH, "processed"))

for path in [PRODATAPATH]:
    if not os.path.exists(path):
        os.mkdir(path)
        

def train_validation_split(df, id_column, train_ratio=0.7, random_state=None):
    """
    Splits a DataFrame into train and validation sets, ensuring that 70% of rows
    for each unique id are in the training set, and the rest in the validation set.

    Parameters:
    - df (pd.DataFrame): The DataFrame to split.
    - id_column (str): The name of the column containing the unique id.
    - train_ratio (float): The ratio of the data to be used for training (default 0.7).
    - random_state (int): Seed for random number generator (default None).

    Returns:
    - train_df (pd.DataFrame): The training DataFrame.
    - val_df (pd.DataFrame): The validation DataFrame.
    """
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame.")

    # Shuffle the DataFrame to ensure randomness
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Initialize empty lists for train and validation data
    train_data = []
    val_data = []

    # Group by the id_column and split each group
    grouped = df.groupby(id_column)
    for _, group in grouped:
        split_point = int(len(group) * train_ratio)
        train_data.append(group.iloc[:split_point])
        val_data.append(group.iloc[split_point:])

    # Concatenate all train and validation data
    train_df = pd.concat(train_data).reset_index(drop=True)
    val_df = pd.concat(val_data).reset_index(drop=True)

    return train_df, val_df

class SiTunesData(BaseData):
    def __init__(self):
        super(SiTunesData, self).__init__()
        self.train_data_path = "train.parquet"
        self.val_data_path = "test.parquet"
        self.num_users = 30
        # self.label_encoding_trained = False
        self.lbe_item= None
        self.lbe_user=None
    def get_num_users(self):
        return self.num_users
    def get_num_items(self):
        return self.load_item_feat()['item_id'].nunique()
    def get_train_data(self):
        return self.get_df('train')

    def get_val_data(self):
        return self.get_df('val')
    def get_features(self, is_userinfo=True):
        user_features = ["user_id"]
        if not is_userinfo:
            user_features = ["user_id"]
        item_features = ['item_id', 'popularity', 'loudness', 'danceability','energy', 'key', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'general_genre_id', 'duration', 'F0final_sma_amean',  'F0final_sma_stddev', 'audspec_lengthL1norm_sma_stddev', 'pcm_RMSenergy_sma_stddev', 'pcm_fftMag_psySharpness_sma_amean',     'pcm_fftMag_psySharpness_sma_stddev', 'pcm_zcr_sma_amean', 'pcm_zcr_sma_stddev']
        reward_features = ["rating"]
        return user_features, item_features, reward_features
        
    def get_df(self, name="train"):
        
        
        interactions_stage1_df = pd.read_csv(DATAPATH / "Stage1"/ "interactions.csv")
        interactions_stage2_df = pd.read_csv(DATAPATH / "Stage2"/ "interactions.csv")
        interactions_stage3_df = pd.read_csv(DATAPATH / "Stage3"/ "interactions.csv")
        
        df_item = self.load_item_feat()
        df_user = self.load_user_feat()
        df_data = pd.concat([interactions_stage1_df,interactions_stage2_df,interactions_stage3_df],axis=0)
        train_df, val_df = train_validation_split(df_data, 'item_id', train_ratio=0.7, random_state=1)
        if name == 'train':
            df_data = train_df
        elif name == 'all':
            pass
        elif name == 'val':
            df_data = val_df
        # if name == 'train':
        #     df_data = pd.concat([interactions_stage1_df,interactions_stage2_df],axis=0)
        # elif name == 'all':
        #     df_data = pd.concat([interactions_stage1_df,interactions_stage2_df,interactions_stage3_df],axis=0)
        # elif name == 'val':
        #     df_data = pd.concat([interactions_stage3_df],axis=0)
        df_data['user_id'] = df_data['user_id']-1
        # if not self.lbe_user:
        #     self.lbe_user = LabelEncoder()
        #     self.lbe_user.fit(df_data['user_id'].unique())
        df_data['item_id'] = self.lbe_item.transform(df_data['item_id'])
        # df_data['user_id'] = self.lbe_user.transform(df_data['user_id'])
        df_data['rating']=df_data['rating'].apply(int)
        # print(df_data['user_id'])
        df_data = df_data.join(df_item,on='item_id', rsuffix='right')
        # print(df_data['user_id'].describe())
        # read interaction
        # filename = os.path.join(PRODATAPATH, name)
        # df_data = pd.read_parquet(filename)
        # df_data = pd.DataFrame([], columns=["user_id", "item_id", "rating"])

        # for item in mat_train.columns:
        #     one_item = mat_train.loc[mat_train[item] > 0, item].reset_index().rename(
        #         columns={"index": "user_id", item: "rating"})
        #     one_item["item_id"] = item
        #     df_data = pd.concat([df_data, one_item])
        # df_data.reset_index(drop=True, inplace=True)

        # read user feature


        # read item features

        # df_data = df_data.join(df_user, on="user_id", how='left')
        # df_data = df_data.join(df_item, on="item_id", how='left')

        # df_data = df_data.astype(int)
        list_feat = None

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        df_data, _, df_item, _ = self.get_df("train")
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
            mat = self.load_mat()
            mat_distance = SiTunesData.get_saved_distance_mat(mat, PRODATAPATH)
            item_similarity = 1 / (mat_distance + 1)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
      
    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            df_data, df_user, df_item, list_feat = self.get_df("all")

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
        # if not self.lbe_user:
        #     self.lbe_user = LabelEncoder()
        #     self.lbe_user.fit(df_user['user_id'].unique())
        # df_user['user_id'] =self.lbe_item.transform(df_user['user_id'])
        return df_user

    def load_item_feat(self):
        df_item = pd.read_csv(DATAPATH / "music_metadata"/ "music_info.csv")
        df_item = df_item.loc[~df_item['item_id'].isin([-1])]
        df_item = df_item.sort_values('item_id').reset_index(drop=True)[['item_id', 'popularity', 'loudness', 'danceability','energy', 'key', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'general_genre_id', 'duration', 'F0final_sma_amean',  'F0final_sma_stddev', 'audspec_lengthL1norm_sma_stddev', 'pcm_RMSenergy_sma_stddev', 'pcm_fftMag_psySharpness_sma_amean',     'pcm_fftMag_psySharpness_sma_stddev', 'pcm_zcr_sma_amean', 'pcm_zcr_sma_stddev']]
        if not self.lbe_item:
            # df_data, _, _, _ = self.get_df('all')
            # df_data = pd.concat([interactions_stage1_df,interactions_stage2_df,interactions_stage3_df],axis=0)
            # df_data['item_id']
            self.lbe_item = LabelEncoder()
            self.lbe_item.fit(df_item['item_id'].unique())

        df_item['item_id'] =self.lbe_item.transform(df_item['item_id'])
        return df_item
    
    # @staticmethod
    def load_mat(self):
        # df_user = pd.DataFrame(np.arange(self.num_users), columns=["user_id"])
        # df_user.set_index("user_id", inplace=True)
        
        interactions_path = os.path.join(PRODATAPATH, "interactions.parquet")
        df_small, df_user, df_item, list_feat = self.get_df('all')
        # df_small['watch_ratio'][df_small['watch_ratio'] > 5] = 5
        # df_small.loc[df_small['rating'] > 5, 'watch_ratio'] = 5

        # lbe_item = LabelEncoder()
        # lbe_item.fit(df_small['item_id'].unique())

        # lbe_user = LabelEncoder()
        # lbe_user.fit(df_small['user_id'].unique())
        # print(df_small['item_id'])
        
        # print(df_small)
        
        # print(df_small['item_id'])
        print((df_small['user_id'].nunique(), df_small['item_id'].nunique()))
        print((df_small['user_id'].max(), df_small['item_id'].max()))
        print((df_small['user_id'].min(), df_small['item_id'].min()))
        print((self.get_num_users(), self.get_num_items()))
        mat = csr_matrix(
            (df_small['rating'],(df_small['user_id'], df_small['item_id'])),
            shape=(self.get_num_users(), self.get_num_items())).toarray()

        return mat
 
if __name__ == "__main__":
    dataset = SiTunesData()
    df_train, df_user_train, df_item_train, _ = dataset.get_train_data()
    df_val, df_user_val, df_item_val, _ = dataset.get_val_data()
    print("SiTunes: Train #user={}  #item={}  #inter={}".format(df_train['user_id'].nunique(), df_train['item_id'].nunique(), len(df_train)))
    print("SiTunes: Test  #user={}  #item={}  #inter={}".format(df_val['user_id'].nunique(), df_val['item_id'].nunique(), len(df_val)))
    # print(df_user_train)
    # print(df_item_train)
    # print(df_train)
    # print('=====================dataset.get_features()')
    # print(dataset.get_features())
    # print('dataset.get_df()')
    # print(dataset.get_df())
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
    
    print('train')
    print(dataset.get_train_data()[0]['item_id'].max())
    print('val')
    print((dataset.get_val_data()[0]['item_id']==940).sum())
    
    
    print(dataset.get_train_data()[0].shape)

    print(dataset.get_val_data()[0].shape)


