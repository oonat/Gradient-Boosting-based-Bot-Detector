import lightgbm as lgbm
from sklearn.model_selection import KFold, train_test_split

from dataset import Dataset
from model import BotClassifier

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)



def run_grid_search(X_train, y_train):
	model = BotClassifier()

	param_grid = {
		'learning_rate': [0.05, 0.1, 1],
#		'n_estimators': [40, 100, 200, 500],
#		'num_leaves': [10, 30, 50, 70, 100],
		'max_depth': [3, 6, 12]
	}

	model.grid_search(X_train, y_train, param_grid)



def cross_validate(cat_ids, X_train, y_train, fold_num):
	kf = KFold(n_splits=fold_num, random_state=1234, shuffle=True)
	folds = kf.split(X_train, y_train)

	for idx, (train_idx, test_idx) in enumerate(folds):
		print("#" * 8 + f" Fold {idx} " + 8 * "#")

		X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
		y_tr, y_val = y_train[train_idx], y_train[test_idx]
		eval_set = [(X_val, y_val)]


		callbacks = [
			lgbm.early_stopping(stopping_rounds=100),
			lgbm.log_evaluation(period=-1),
		]

		model = BotClassifier()
		model.train(
			X_tr,
			y_tr,
			cat_features=cat_ids,
			eval_set=eval_set,
			callbacks=callbacks,
		)

		score = model.evaluate(X_val, y_val)
		model.plot_feature_importance("Fold" + str(idx) + "_plot.png")
		print(f"Fold {idx} F1-score: {score:.5f}\n")




def test(cat_ids, X_train, y_train, X_test, y_test):
	model = BotClassifier()
	eval_set = [(X_test, y_test)]

	callbacks = [
		lgbm.early_stopping(stopping_rounds=100),
		lgbm.log_evaluation(period=-1),
	]

	model.train(
		X_train,
		y_train,
		cat_features=cat_ids,
		eval_set=eval_set,
		callbacks=callbacks,
	)

	score = model.evaluate(X_test, y_test)
	print(f"Test set F1-score: {score:.5f}.\n")




def main():
	with open('datasets/usernames.txt', 'r') as f:
		username_list = [line.strip() for line in f]

	dataset_list = [
		'botometer-feedback-2019',
		'botwiki-2019',
		'celebrity-2019',
		'cresci-rtbust-2019',
		'cresci-stock-2018',
		'gilani-2017',
		'political-bots-2019',
		'pronbots-2019',
		'vendor-purchased-2019',
		'verified-2019',
	]

	data = Dataset(username_list=username_list, path_list=dataset_list)

	X_train, X_test, y_train, y_test = \
		train_test_split(data.X, data.y, 
		train_size=0.8, random_state=1234, shuffle=True)


	#run_grid_search(X_train, y_train)

	#cross_validate(data.cat_ids, X_train, y_train, fold_num=5)
	#test(data.cat_ids, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
	main()