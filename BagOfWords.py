import re
import os.path
import numpy as np
import utils
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import words
from nltk.stem import PorterStemmer
 
# init stemmer
porter_stemmer=PorterStemmer()

ignore = set(stopwords.words('english'))
eng_words = set(nltk.corpus.words.words())

def word_extraction(sentence):

	#print(ignore)
	words = re.sub("[^\w]", " ", sentence).split()
	cleaned_text = [w.lower() for w in words if (w not in ignore and w in eng_words)]

	return cleaned_text


def tokenize(sentences):

	words = []
	for sentence in sentences:
		w = word_extraction(sentence)
		words.extend(w)
		words = sorted(list(words))

	print(words)
	return words

 
def preprocessor(text):

	text = re.sub("[^\w]", " ", text).split()
	text = [w.lower() for w in text if (w not in ignore and w in eng_words)]
	stemmed_text=[porter_stemmer.stem(word=w) for w in text]

	return ' '.join(stemmed_text)


def get_sentiment(ratings):
	# Takes dataframe of ratings

	df_sentiment = pd.DataFrame(0, index=np.arange(len(ratings.to_numpy())), columns=['sentiment'])

	df_sentiment['sentiment'].loc[ratings >= 7.0] = 1.0
	df_sentiment['sentiment'].loc[ratings < 7.0] = 0.0

	return df_sentiment['sentiment']



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

def get_all_comments(infile):

	try:
		df = pd.read_csv(infile)

		allcomments = []
		for comment in df['comment']:
			allcomments.append(comment)

		return allcomments

	except pd.errors.ParserError:
		print('Need CSV file')


def run_binary_class_model(comments_df):

	from sklearn.model_selection import train_test_split
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Dropout

	X_train, X_test, Y_train, Y_test = train_test_split(comments_df['array'].values, comments_df['sentiment'].values, test_size=0.20, random_state=42)

	print(Y_test)
	print(X_test.shape)

	n_words = X_test.shape[1]

	# define network
	model = Sequential()
	model.add(Dense(50, input_shape=(218,), activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# compile network
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, Y_train, epochs=50, verbose=2)
	# evaluate
	loss, acc = model.evaluate(X_test, Y_test, verbose=0)
	scores.append(acc)
	print('accuracy: %s' % acc)
	return scores

bgg_file = '/home/josh/Documents/Code/Projects/bgg/board-game-scraper/feeds/bgg/GameItem/2020-05-27T05-38-16.jl'
user_file = '/home/josh/Documents/Code/Projects/board-game-filler/test.jl'
user_data = pd.read_json(user_file, lines=True)
bgg_data = pd.read_json(bgg_file, lines=True)

bgg_data = bgg_data.set_index('bgg_id')

allcomments = user_data[user_data['comment'].notna()]['comment'] 
allratings = user_data[user_data['comment'].notna()]['bgg_user_rating']
allbggids = user_data[user_data['comment'].notna()]['bgg_id']

comments_df = pd.DataFrame({'comment': allcomments.values, 'rating': allratings.values, 'bgg_id': allbggids.values})#.sort_values(by='bgg_id')

del allcomments 
del allratings 
del allbggids

#bg = pd.DataFrame()
#print(bgg_data)

#for i in comments_df['bgg_id'].values:

#	try:
#		bg = bg.append(bgg_data.loc[i,:])
#
#	except KeyError:
#		continue

#print(bg['name'])


#print(utils.get_bgg_from_user(allbggids, bgg_data))


#print(comments_df['comment'])

#allcomments = get_all_comments("test.csv")

#print(len(tokenize(allcomments)))


from sklearn.feature_extraction.text import CountVectorizer

#print(comments_df['comment'][0:10].values)

comments_reduced = comments_df.iloc[0:1000]

vectorizer = CountVectorizer(min_df=0.01, preprocessor=preprocessor)
X = vectorizer.fit_transform(comments_reduced['comment'].values)
print(X.toarray().shape)
#print(X.shape)
print(vectorizer.get_feature_names())

comments_reduced['sentiment'] = get_sentiment(comments_reduced['rating']).values

word_count = pd.DataFrame({'col':[z for z in X.toarray()]})


comments_reduced['array'] = word_count['col'].values

#print(comments_reduced['array'])

#print(comments_reduced['sentiment'])

run_binary_class_model(comments_reduced)