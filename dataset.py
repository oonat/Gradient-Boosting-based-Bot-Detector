import pandas as pd
import numpy as np
from utils import *


class Dataset:
	def __init__(self, username_list, json_path = None, csv_path = None, path_list = None):
		if path_list:
			self.frame = self._merge_datasets(path_list)
		else:
			self.frame = self._load_dataset(json_path, csv_path)

		self.feature_list = [
				'statuses_count',
				'followers_count',
				'friends_count',
				'favourites_count',
				'listed_count',
				'default_profile',
				'profile_use_background_image',
				'verified',
				'tweet_freq',
				'followers_growth_rate',
				'friends_growth_rate',
				'favourites_growth_rate',
				'listed_growth_rate',
				'followers_friends_ratio',
				'screen_name_length',
				'description_length',
				'screen_name_likelihood',
				'label',
			]

		self.lm = create_language_model(username_list)

		self.frame = self._process_frame(self.frame)

		self.X, self.y = self.frame.drop(['label'], axis=1), self.frame[['label']].values.flatten().astype('int')

		print(f'# of Bots: {sum(self.y)}')
		print(f'# of Humans: {len(self.y) - sum(self.y)}')

		cat_features = self.X.select_dtypes(exclude=np.number).columns.to_list()

		for col in cat_features:
			self.X[col] = pd.Categorical(self.X[col])

		self.cat_ids = [self.X.columns.get_loc(col) for col in cat_features]


	def _load_dataset(self, json_path, csv_path):
		df_tweet = pd.read_json(json_path)
		df_tweet_user = pd.json_normalize(df_tweet.user)

		df_tweet_user['user_age'] = df_tweet.apply(lambda x: calculate_age(x['created_at'], x.user['created_at']), axis=1)

		df_labels = pd.read_csv(csv_path, sep="\t", names=["id", "label"])
		df_joined = pd.merge(df_tweet_user, df_labels, on='id', how='inner')

		df_joined.loc[df_joined.label == "bot", 'label'] = 1
		df_joined.loc[df_joined.label == "human", 'label'] = 0

		return df_joined


	def _merge_datasets(self, path_list):
		print(path_list[0])
		df_all = self._load_dataset("datasets/" + path_list[0] + "_tweets.json", "datasets/" + path_list[0] + ".tsv")

		for i in path_list[1:]:
			print(i)
			df = self._load_dataset("datasets/" + i + "_tweets.json", "datasets/" + i + ".tsv")
			df_all = pd.concat([df_all, df], ignore_index=True)
		
		return df_all


	def _process_frame(self, frame):
		# remove duplicate rows
		frame = frame.drop_duplicates(subset='id', keep="last").reset_index(drop=True)

		frame['tweet_freq'] \
			= frame.apply(lambda x: (x['statuses_count'] / max(x['user_age'], 1)), axis=1)

		frame['followers_growth_rate'] \
			= frame.apply(lambda x: (x['followers_count'] / max(x['user_age'], 1)), axis=1)

		frame['friends_growth_rate'] \
			= frame.apply(lambda x: (x['friends_count'] / max(x['user_age'], 1)), axis=1)

		frame['favourites_growth_rate'] \
			= frame.apply(lambda x: (x['favourites_count'] / max(x['user_age'], 1)), axis=1)

		frame['listed_growth_rate'] \
			= frame.apply(lambda x: (x['listed_count'] / max(x['user_age'], 1)), axis=1)

		frame['followers_friends_ratio'] \
			= frame.apply(lambda x: (x['followers_count'] / max(x['friends_count'], 1)), axis=1)

		frame['screen_name_length'] = frame.apply(lambda x: len(x['screen_name']), axis=1)
		frame['description_length'] = frame.apply(lambda x: len(x['description']), axis=1)
		frame['screen_name_likelihood'] = frame.apply(lambda x: calculate_likelihood(self.lm, x['screen_name']), axis=1)

		return frame[self.feature_list]
