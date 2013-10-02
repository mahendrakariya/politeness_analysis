import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

def load_data():
	""" Loads the pickled dataset and returns only the request column """
	wiki_polite = pickle.load(open('Stanford_politeness_corpus/wikipedia.annotated.polite.pickle'))
	wiki_impolite = pickle.load(open('Stanford_politeness_corpus/wikipedia.annotated.impolite.pickle'))
	se_polite = pickle.load(open('Stanford_politeness_corpus/stack-exchange.annotated.polite.pickle'))
	se_impolite = pickle.load(open('Stanford_politeness_corpus/stack-exchange.annotated.impolite.pickle'))
	del wiki_polite['Normalized Score']
	del wiki_impolite['Normalized Score']
	del se_polite['Normalized Score']
	del se_impolite['Normalized Score']
	#logging.debug("All 4 datasets loaded")
	return wiki_polite, wiki_impolite, se_polite, se_impolite

def calc_accuracy(predictions, expectations):
	""" Calculates the accuracy of model """
	#logging.debug("Calculating Accuracy... Please wait...")
	total_predictions = float(len(predictions))
	correct_predictions = 0.0
	for i in range(len(predictions)):
		if predictions[i] == expectations[i]:
			correct_predictions += 1.0
	print "Accuracy of classifier = %f" % (correct_predictions / total_predictions)

def train_and_predict():
	""" Vectorizes the data, trains the classifier and finally predicts the class for test data """
	wiki_polite, wiki_impolite, se_polite, se_impolite = load_data()
	
	wiki = wiki_polite.append(wiki_impolite)
	se = se_polite.append(se_impolite)
	
	complete_set = wiki.append(se)

	vectorizer = CountVectorizer()

	vectorizer.fit(complete_set['Request'])
	test_set = vectorizer.transform(wiki['Request'])
	train_set = vectorizer.transform(se['Request'])

	test_labels = [1]*len(wiki_polite) + [-1]*len(wiki_impolite)
	train_labels = [1]*len(se_polite) + [-1]*len(se_impolite)

	clf = LinearSVC(C=0.092)
	clf.fit(train_set, train_labels)
	predictions = clf.predict(test_set)
	calc_accuracy(predictions, test_labels)

def main():
	train_and_predict()

if __name__ == '__main__':
	main()
