import lightgbm as lgbm
from sklearn.metrics import log_loss, accuracy_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from .preprocessor import Preprocessor
import joblib


class BotClassifier:
	def __init__(self, lr=0.01, n_estimators=10000, num_leaves=100):
		self.clf = lgbm.LGBMClassifier(objective='binary', 
			learning_rate=lr,
			n_estimators=n_estimators,
			num_leaves=num_leaves)


	def train(self, X_train, y_train, eval_set, cat_features, callbacks, metric="binary_logloss"):
		self.clf.fit(
			X_train,
			y_train,
			eval_set=eval_set,
			categorical_feature=cat_features,
			eval_metric=metric,
			callbacks=callbacks,
		)


	def evaluate(self, X_eval, y_eval):
		preds = self.clf.predict(X_eval)
		print("Confusion matrix: \n", confusion_matrix(y_eval, preds))
		print(f"Accuracy: {accuracy_score(y_eval, preds)}")
		return f1_score(y_eval, preds)


	def grid_search(self, X_train, y_train, param_grid):
		gbm = GridSearchCV(self.clf, param_grid, cv=5)
		gbm.fit(X_train, y_train)	
		print(f'Best parameters: {gbm.best_params_}')


	def plot_feature_importance(self, fname="feat_importance.png"):
		ax = lgbm.plot_importance(self.clf, max_num_features=None, figsize=(20,15))
		plt.savefig(fname)


	def save(self, path):
		joblib.dump(self.clf, path)


	def load(self, path):
		self.clf = joblib.load(path)


	def classify(self, json_list):
		processor = Preprocessor()
		frame = processor.process(json_list)

		return self.clf.predict(frame)