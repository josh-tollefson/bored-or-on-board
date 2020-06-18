import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter_stemmer=PorterStemmer()
Lem = WordNetLemmatizer()
ignore = set(stopwords.words('english'))
eng_words = set(nltk.corpus.words.words())

def word_extraction(sentence):

	words = re.sub("[^\w]", " ", sentence).split()
	cleaned_text = [w.lower() for w in words if (w not in ignore and w in eng_words)]

	return cleaned_text

def tokenize(sentences):

	words = []
	for sentence in sentences:
		w = word_extraction(sentence)
		words.extend(w)
		words = sorted(list(words))

	return words

def preprocessor(text, labels=True):

	text = re.sub("[^\w]", " ", text).split()
	text = [Lem.lemmatize(w.lower()) for w in text]
	if labels:
		text = [w.lower() for w in text if ((w not in ignore and w in eng_words and w not in ['star', 'play', 'game']) or (w in ['filler', 'bits', 'miniature', 'mini', 'minis']))]
		text = [porter_stemmer.stem(word=w) for w in text]
		return ' '.join(text)

	else:
		text = [w.lower() for w in text if ((w not in ignore and w in eng_words) or (w in ['filler', 'bits', 'miniature', 'mini', 'minis']))]
		text = [porter_stemmer.stem(word=w) for w in text]
		return  ' '.join(text)

def preprocess_exp_stopwords(text, labels=False):

	mechanics_sentence = "competitive interaction strategy tactics replay gameplay mechanic system elimination random decision tense deduction luck scoring dynamics"
	theme_sentence = 'art theme beautiful colorful design atmospheric universe'
	component_sentence = 'component layout marker tile box small large wear insert worn miniature minis production'
	length_sentence = 'slow quick fast filler setup takedown hours long'
	accessible_sentence = 'rule family learn easy hard difficult difficulty disability weight complex complicated accessible simple'


	mechanics_list = mechanics_sentence.split(' ')
	theme_list = theme_sentence.split(' ')
	component_list = component_sentence.split(' ')
	length_list = length_sentence.split(' ')
	accessible_list = accessible_sentence.split(' ')

	text = re.sub("[^\w]", " ", text).split()
	text = [Lem.lemmatize(w.lower()) for w in text]
	if labels:
		text = [w.lower() for w in text if ((w not in ignore and w in eng_words and w not in ['star', 'play', 'game']) or (w in ['filler', 'bits', 'miniature', 'mini', 'minis']))]
		text = [porter_stemmer.stem(word=w) for w in text]
		return stemmed_text

	else:
		text = [w.lower() for w in text if ((w not in ignore and w in eng_words and w not in theme_list) or (w in ['filler', 'bits', 'miniature', 'mini', 'minis']))]
		text = [porter_stemmer.stem(word=w) for w in text]
		return  ' '.join(text)
