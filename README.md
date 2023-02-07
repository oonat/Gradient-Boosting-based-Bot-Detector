
# Introduction
Bot detector is a gradient-boosting-based classifier implemented as a part of my graduation project named "Skepsis". The idea and implementation are mainly based on a paper named "Scalable and Generalizable Social Bot Detection through Data Selection." (Yang et al., 2020)

The model uses the features given below:
- statuses_count
- followers_count
- friends_count
- favourites_count
- listed_count
- default_profile
- profile_use_background_image
- verified
- tweet_freq
- followers_growth_rate
- friends_growth_rate
- favourites_growth_rate
- listed_growth_rate
- followers_friends_ratio
- screen_name_length
- description_length

# How to Use

## Installation

Before installing the required libraries, It is recommended to create a virtual environment.

The libraries required for the project are listed in the **requirements.txt** file. To download and install the necessary libraries,
```sh
pip install -r requirements.txt
```

## Model Training
The python files required for model training can be found in **model_training** folder. So change your active directory to **model_training** before applying the steps given below.


The model was trained using the datasets given at https://botometer.osome.iu.edu/bot-repository/datasets.html. The names of the datasets used for training are listed below:

- botometer-feedback-2019
- botwiki-2019
- celebrity-2019
- cresci-rtbust-2019
- cresci-stock-2018
- gilani-2017
- political-bots-2019
- pronbots-2019
- vendor-purchased-2019
- verified-2019

Due to legal concerns, the datasets used to train the model were not added to this repo.

Before the model training, please create a folder named **datasets** and install the datasets listed above into this folder.

To load these datasets,

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

	data = Dataset(path_list=dataset_list)

A simple code block to train and save the model is given below,

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
	
	model.save("saved_model.pkl")

Useful functions for training and testing of the models and example usages can be seen in **main.py** file.

## Bot Classification
After training the model, one can use it to classify Twitter accounts as bot or not. Bot classifier requires the profile features provided by Twitter in JSON format. To obtain these features, Twitter API can be used.

    model = BotClassifier()
    model.load("saved_model.pkl")
    predicted_labels = model.classify(data)


# References
Yang, K.-C., Varol, O., Hui, P.-M., & Menczer, F. (2020). Scalable and generalizable social bot detection through data selection. _Proceedings of the AAAI Conference on Artificial Intelligence_, _34_(01), 1096–1103. https://doi.org/10.1609/aaai.v34i01.5460
