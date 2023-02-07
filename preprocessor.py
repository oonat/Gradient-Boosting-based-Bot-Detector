import pandas as pd
import numpy as np
from .utils import *

class Preprocessor:
	def __init__(self, feature_list = None):

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
			] if feature_list is None else feature_list



	def process(self, json_list):
		frame = pd.DataFrame(json_list)
		frame = pd.json_normalize(frame.user)

		# remove duplicate rows
		frame.drop_duplicates(subset='id', keep="last").reset_index(drop=True)

		current_time = datetime.datetime.now(datetime.timezone.utc)
		frame['user_age'] = frame.apply(lambda x: calculate_age(current_time, x['created_at']), axis=1)

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

		frame = frame[self.feature_list]

		cat_features = frame.select_dtypes(exclude=np.number).columns.to_list()

		for col in cat_features:
			frame[col] = pd.Categorical(frame[col])
		
		return frame