import re
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import utils
import wordprocess as wp
import timeit

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier, XGBRFClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression



def freq_words(allcomments):

	all_words = ' '.join([text for text in allcomments])
	all_words = all_words.split()
	fdist = nltk.FreqDist(all_words)
	words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

	d = words_df.nlargest(columns="count", n = 20)


	plt.figure(figsize=(12,15))
	ax = sns.barplot(data=d, x="count", y="word")
	ax.set(ylabel = 'word')
	plt.show()




def extract_topn_from_vector(feature_names, sorted_items, topn=50):
    """return n-gram counts in descending order of counts"""
    
    sorted_items = sorted_items[:topn]
 
    results=[]
    
    for idx, count in sorted_items:
        n_gram=feature_names[idx]
        results.append((n_gram,count))
 
    return results



def run_binary_class_model(comments_df):

	from sklearn.model_selection import train_test_split
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Dropout

	X_train, X_test, Y_train, Y_test = train_test_split(comments_df.iloc[:,4:], comments_df.iloc[:,3], test_size=0.20, random_state=42)

	n_words = X_test.shape[1]

	#run_pca(X_test, X_train)

	# define network
	model = Sequential()
	model.add(Dense(300, input_shape=(n_words,), activation='relu'))
	model.add(Dense(100, input_shape=(n_words,), activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# compile network
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test), verbose=2)
	# evaluate
	loss, acc = model.evaluate(X_test, Y_test, verbose=0)
	#scores.append(acc)
	print('accuracy: %s' % acc)
	predictions = model.predict(X_test)

	print ("\nclassification report :\n",(confusion_matrix(Y_test,np.round(predictions))))

	from sklearn.svm import SVC
	SVM_model=SVC()
	SVM_model.fit(X_train,Y_train)


	#scores.append(acc)
	predictions = SVM_model.predict(X_test)
	print(predictions)

	print ("\nclassification report :\n",(confusion_matrix(Y_test,np.round(predictions))))


	return model

def run_pca(X_test, X_train):

	from sklearn.decomposition import PCA
	pca = PCA() 
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	total=sum(pca.explained_variance_)
	k=0
	current_variance=0
	while current_variance/total < 0.99:
		current_variance += pca.explained_variance_[k]
		k=k+1

	print(k)

	pca = PCA(n_components=k)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	plt.figure()
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('Number of Components')
	plt.ylabel('Variance (%)') #for each component
	plt.title('Exoplanet Dataset Explained Variance')
	plt.show()


def run_model(model, X, Y, kfolds = None):


	zero_label = list(Y).count(0)
	one_label = list(Y).count(1)
	baseline_acc = (zero_label / (zero_label + one_label))**2 + (one_label / (zero_label + one_label))**2
	baseline_pre = one_label*one_label / (zero_label*one_label + one_label*one_label)
	print(baseline_pre)


	scores = []
	models = []

	start = timeit.default_timer()

	if kfolds is None:

		from sklearn.model_selection import train_test_split

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
		sm = SMOTE(1.0)
		X_train, Y_train = sm.fit_sample(X_train, Y_train.ravel())
		model.fit(X_train, Y_train)
		scores.append(model.score(X_test, Y_test))
		models.append(model)

	else:

		for train_index, test_index in kfolds.split(X, Y):

		    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
		    sm = SMOTE(1.0)
		    X_train, Y_train = sm.fit_sample(X_train, Y_train.ravel())
		    model.fit(X_train, Y_train)
		    scores.append(model.score(X_test, Y_test))
		    models.append(model)

	stop = timeit.default_timer()

	runtime = stop - start


	Y_pred = model.predict(X_test)
	stddev = np.std(scores)

	report = classification_report(Y_test, Y_pred, output_dict=True)
	return models, scores, report, runtime, baseline_pre, stddev


###########################

nmax = 30000
max_features = 5000
category_tag = 'theme'


FOLDER = './'
INPUT_PANDAS = 'bgg_psuedolabels.csv'
INPUT_PANDAS = 'hand_labeled_copy.csv'

FOLDER_SAVE = './save/'
OUTPUT_MODEL = 'model_tfid_category_mnb_hand_1_3_ncomments' + str(nmax) + '_nfeatures' + str(max_features) + '_' + category_tag + '.sav' 
OUTPUT_VZER = 'vzer_tfid_category_mnb_hand_1_3_ncomments' + str(nmax) + '_nfeatures' + str(max_features) + '_' + category_tag + '.sav' 
OUTPUT_REPORT = 'report_tfid_category_mnb_hand_1_3_ncomments' + str(nmax) + '_nfeatures' + str(max_features) + '_' + category_tag + '.csv'

df = utils.load_dataframe(FOLDER, INPUT_PANDAS)
df = df.fillna(0)

comments_reduced = df.iloc[:nmax]


# model = RandomForestClassifier(n_estimators=100, 
#                                bootstrap = True,
#                                max_features = 'sqrt')

#model = XGBRFClassifier()
#model = LogisticRegression()
model = MultinomialNB()

############################


#############################

Tfmer = TfidfVectorizer(sublinear_tf=True, max_features=max_features, ngram_range=(1, 3), preprocessor=wp.preprocessor)
X_transformed = Tfmer.fit_transform(comments_reduced['comment'].values)
Y = comments_reduced[category_tag].values

kf = StratifiedKFold(n_splits=5, shuffle=True) 

models, scores, report, runtime, baseline_stats, stddev = run_model(model, X_transformed, Y, kfolds=kf)

print(report)
print(max(scores))
print(np.std(scores))

utils.save_model(models[np.squeeze(np.where(scores == max(scores))[0])], FOLDER_SAVE, OUTPUT_MODEL)
utils.save_model(Tfmer, FOLDER_SAVE, OUTPUT_VZER)
utils.save_report(report, FOLDER_SAVE, OUTPUT_REPORT, runtime=runtime, baseline=baseline_stats, stddev=stddev)
