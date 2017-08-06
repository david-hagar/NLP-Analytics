# NLP-Analytics
Code examples for open source NLP analytics and lists of resources

# Resources

1. Java
    1. [OpenNLP](https://opennlp.apache.org/)
2. Python
    1. [scikit-learn](http://scikit-learn.org/stable/)
    4. [TFIDF](http://scikit-learn.org/stable/modules/feature_extraction.html)
    5. [nltk](http://www.nltk.org/)
    5. [spacy.io](https://spacy.io) - part of speech and entity extraction
    6. [Gensim](https://radimrehurek.com/gensim/) - tfidf, word2vec and others
7. C
    1. [Senna](https://ronan.collobert.com/senna/)
9. DataSets
    1. [nltk datasets](http://www.nltk.org/nltk_data/) 


# Bag Of Words Feature Extraction

1. Clean text of noise content (Ex. email headers and signatures, non-text documents, bad html tags)
1. Tokenize each document into a list of features and feature counts. Features can be:
    1. sequence of non-whitespace chatacters (most common)
    3. character n-grams ( Ex. "Hello world" -> "he" "el" "ll" "lo" "o " " w" "wo" ...)
    4. word n-grams (Ex. "The brown fox" -> "the" "the brown" "brown" "brown fox")
    5. grammar parsed noun and verb phrases plus words
    5. word hashes modulo N
6. Optional word transforms
    1. remove stopwords
    8. [stem](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)  ("walks" -> "walk", "walking" -> "walk" )
    9. lematization ("are" -> "be",  "is" -> "be") 
6. Assemble a global document count for each word 
7. Form vocabulary from document frequencies bettween 90% of documents to 2 documents.
8. Build [TDIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) wieghts from doc counts
9. Multiply document counts by TFIDF weight and optionally normalize to Euclidian length of 1.
10. Plug weighted counts into your favorite machine learning algorithm


# Common Bag of Words Analytics
1. K-means Clustering
2. Classification
    1. [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
    2. [SVM](https://en.wikipedia.org/wiki/Support_vector_machine)
    3. [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
10. Word Vectors - Converts sparce word counts into dense vector representing "context" of word
    1. [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
    4. [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
    5. [GloVe](https://nlp.stanford.edu/projects/glove/)
5. Topic Modeling - identifies multiple "topic vectors" that sum in different amounts to for each document in the corpus.
    1. [Latent Dirichlet allocation](Latent Dirichlet allocation)


# Running Jupyter Notebook Examples
1. install python (recomend python 3)
    1.   on mac: `brew install python3`
2. `pip3 install jupyter`
3. install any prerequisite packages (Ex. `pip3 install sklearn` 
3. cd to directory where *.ipynb files are
4. `jupyter notebook`
5. this should open a web browser where you can launch each notebook
6. "shift-return" runs each command
7. `ctrl-c` from command line exits

Examples in the repo:
1. K-means clustering of movie subtitles with sci-kit learn. [(link)](https://github.com/david-hagar/NLP-Analytics/tree/master/python/sklearn)
