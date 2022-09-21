# Databricks notebook source
sc # checkign spark context

# COMMAND ----------

# Part 1 - Word count for named entities
# Getting things setup to be able to utilize spaCy NLP library for named entity extraction
dbutils.library.installPyPI("spacy")
import spacy
spacy.cli.download("en")
nlp = spacy.load('en_core_web_sm')

# COMMAND ----------

# import data file and do some initial processing- working with the novel Anna Karenina by Leo Tolstoy
text = sc.wholeTextFiles('/FileStore/tables/AnnaKarenina.txt')
oneString = text.collect()[0][1]
oneString = oneString.replace("\r", "")
oneString = oneString.replace("\n", "")
oneString

# COMMAND ----------

# Adjusting max length for spaCy NLP processing library and feeing input to it for processing
nlp.max_length = 2000000
document = nlp(oneString)

# COMMAND ----------

# Extracting named entities via spaCY NLP library
namedEntArray = []
ents = [e.text for e in document.ents]
for item in document:
  if item.text in ents:
    namedEntArray.append(item.text)

# COMMAND ----------

# converting filtered array containing only named entities to RDD
namedEntRDD = spark.sparkContext.parallelize(namedEntArray)
# removing words with length less than 5 and kaking all lowercase
namedEntRDD = namedEntRDD.flatMap(lambda x : x.split(" ")).filter(lambda x : len(x) > 5).map(lambda x: x.lower())
# performing reduce to get sum of counts
wordCount =  namedEntRDD.map(lambda x : (x, 1)).reduceByKey(lambda x,y: x + y )
# sorting in decending order, most frequent named entities are at top.
# Looks like russian names and locations as expected since the selected text was for the novel Anna Karenina 
wordCount.sortBy(lambda x : -x[1]).collect()

# COMMAND ----------

# Part 2 - Search Engine for Movie Plot Summaries
# Import File first into RDD
# Used a smaller file with first 100 movie plots from full file to improve execution time during development/testing
inputLines = sc.textFile("/FileStore/tables/minimized_plots2.txt")
# switch with line below to use complete file (likely will take much longer to execute commands)
#inputLines = sc.textFile("/FileStore/tables/plot_summaries.txt")

# COMMAND ----------

# Split into RDD in the format of Key = movieID and Value = plot summary for a given movieID
inputLinesSplit = inputLines.map(lambda x: ((x.split("\t"))[0], (x.split("\t"))[1]))
inputLinesSplit.collect()

# COMMAND ----------

# Setup everything necessary to use NLTK for text processing (removing stop words, adjusting for punctuation, spacing, etc)
# BELOW commands do not always work, better to select cluster within compute tab and install nltk library via pypi
# If installing via UI then no need to run this block - inconsistent behavior

#%sh 
#pip install nltk
#pip install --upgrade pip
#python -m nltk.downloader all

# COMMAND ----------

# Split by mapping words to documentIDs (in this case movieIDs) and add one to be able to perform term count for document with reduce
inputLinesFlatMap = inputLinesSplit.flatMap(lambda x: [((x[0],word.lower()),1) for word in x[1].split()])
inputLinesFlatMap.collect()

# COMMAND ----------

# setup to use NLTK for stop word removal
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

# COMMAND ----------

# filter out stop words pulled earlier from NLTK
nltk_stop_words = stopwords.words('english')
inputLinesStopless = inputLinesFlatMap.filter(lambda x : x[0][1] not in nltk_stop_words).filter(lambda x : len(x[0][1]) > 3)
inputLinesStopless.collect()

# COMMAND ----------

# Get total term count for each document
wordsPerMoviePlot = inputLinesStopless.map(lambda x : (x[0][0], x[1])).reduceByKey(lambda x, y: x + y)
wordsPerMoviePlot.collect()

# COMMAND ----------

# Give number of times a given word was found for each document - (Term Frequencies per document)
termInstancesPerDocument = inputLinesStopless.reduceByKey(lambda x,y : x + y)
termInstancesPerDocument.collect()

# COMMAND ----------

# Adjusting RDD to have key available for joining
termInstancesPerDocumentKeyAdjust = termInstancesPerDocument.map(lambda x : ((x[0][0]), (x[0][1], x[1])))
termInstancesPerDocumentKeyAdjust.collect()

# COMMAND ----------

# Adjusting RDD to have key available for joining
inputLinesStoplessKeyAdjust = inputLinesStopless.map(lambda x : (x[0][0],(x[0][1], x[1])))
inputLinesStoplessKeyAdjust.collect()

# COMMAND ----------

# perform join with DocumentID (movie ID) as key 
wordCountDocAndTotal = wordsPerMoviePlot.join(termInstancesPerDocumentKeyAdjust)
# Now I have RDD containing (DocumentID, (term, term count within document), total term count in entire document)
wordCountDocAndTotal.collect()

# COMMAND ----------

# Dividing count of term in each document by the number of total words in the document - this is term frequency adjusted for document  size
TF_termPerDoc = wordCountDocAndTotal.map(lambda x : (x[0], x[1][1][0], (float(x[1][1][1])/x[1][0])))
TF_termPerDoc.collect()

# COMMAND ----------

termInstancesPerDocument.collect()

# COMMAND ----------

# Now I need document frequency - how many times a term is found in a document 
# Adjusting RDD structure to have words be the key and have document number + counts as values
termDocumentInstanceMap = termInstancesPerDocument.map(lambda x : (x[0][1], 1))
termDocumentInstanceMap.collect()

# COMMAND ----------

# Now reduce to get term, number of documents with an instance of term
reducedDocumentFrequency = termDocumentInstanceMap.reduceByKey(lambda x, y : x+y)
reducedDocumentFrequency.collect()

# COMMAND ----------

TF_termPerDoc.collect()

# COMMAND ----------

TF_termPerDoc_keyAdjust = TF_termPerDoc.map(lambda x : ((x[1]), (x[0], x[2])))
TF_termPerDoc_keyAdjust.collect()

# COMMAND ----------

# consolidate RDD to contain term, document frequency, documentID, term frequency, 
consolidatedRDD = reducedDocumentFrequency.join(TF_termPerDoc_keyAdjust)
consolidatedRDD.collect()

# COMMAND ----------

# time to add inverse document frequency : term frequency * log(N/df)
import math
N = inputLinesSplit.count() # This is total document count 
TFIDF = consolidatedRDD.map(lambda x : (x[0], x[1][1][0], x[1][1][1] * math.log10(N/x[1][0])))
TFIDF.collect()            
# term, documentID, TF-IDF

# COMMAND ----------

# Now get RDD with 1 entry per term containing document ID having max IDF value
maxTermTFIDFValues = TFIDF.map(lambda x : (x[0], x)).reduceByKey(lambda x1, x2: max(x1, x2, key=lambda x: x[-1])) 
maxTermTFIDFValues.collect()

# COMMAND ----------

# import file with single words for search
singleSearchWords2 = sc.textFile("/FileStore/tables/singleSearchWords-1.txt")
singleSearchWordsPairRDD = singleSearchWords2.map(lambda x: (x.split(" ")[0], x))
singleSearchWordsPairRDD.collect()

# COMMAND ----------

# perform join between search terms and RDD containing term, Movie ID with max TF-IDF value, and max TF-IDF value
joinedMaxTermSearch = maxTermTFIDFValues.join(singleSearchWordsPairRDD).values()
joinedMaxTermSearch.map(lambda x : (x[0][0], x[0][1], x[0][2])).collect()

# COMMAND ----------

# adjust RDD structure to join with meta data to be able to get movie title
joinedMaxTermSearchKeyAdjust = joinedMaxTermSearch.map(lambda x : (x[0][1], x[0][0]))
joinedMaxTermSearchKeyAdjust.keys().collect()

# COMMAND ----------

# Import movie metadata file to be able to retrieve titles for MovieID
movieMetaData = sc.textFile("/FileStore/tables/movie_metadata-1.tsv")
movieMetaDataSplit = movieMetaData.map(lambda x: ((x.split("\t"))[0],(x.split("\t"))[2]))
movieMetaDataSplit.collect()

# COMMAND ----------

# Here you get movie names with max tf-idf values for each single search word from file
joinWithMetaData = movieMetaDataSplit.join(joinedMaxTermSearchKeyAdjust).map(lambda x : (x[1][1], x[1][0]))
joinWithMetaData.collect()
# This is the result  set for out query - end of part 2 section 1: User enters single term

# COMMAND ----------

multiTermSearchQuery = sc.textFile("/FileStore/tables/multiTermQuery-2")
searchQuery1 = multiTermSearchQuery.take(1)
sc.parallelize(searchQuery1).flatMap(lambda x : x.split(" "))
sc.parallelize(searchQuery1).collect()

searchQueryTerms = sc.parallelize(searchQuery1).flatMap(lambda x : x.split())
searchQueryTerms.collect()

# COMMAND ----------

# filter out stop words from query terms using method applied earlier via NLTK
nltk_stop_words = stopwords.words('english')
searchQueryTermsStopless = searchQueryTerms.filter(lambda x : x not in nltk_stop_words)
searchQueryTermsStopless.collect()

# COMMAND ----------

# Need to get TDF for input query
queryTermCount = searchQueryTermsStopless.count()
searchQueryMapped = searchQueryTermsStopless.map(lambda x : ( (x) , 1))
searchQueryTermsTotalCount = searchQueryMapped.reduceByKey(lambda x , y : x + y)

searchQueryConsolidated = searchQueryTermsTotalCount.map(lambda x : (x[0], (x[1], queryTermCount)))
searchQueryIDF = searchQueryConsolidated.map(lambda x : (x[0], (float(x[1][1])/x[1][0])))

#take Log so IDF values do not increase radically with corpus size
import math # importing although should already be done earlier
# not multiplying by term frequency assuming that terms will not be repeated in search queries
searchQueryIDFNormalized = searchQueryIDF.map(lambda x : (x[0], math.log(x[1]))) 
searchQueryIDFNormalized.collect()

# No we have normalized IDF values for the query itself
# will be used for cosine comparison among documents for multisearch criteria

# COMMAND ----------

searchQueryIDFMap  =  searchQueryIDFNormalized.map(lambda x : (('Query'), (x[0], x[1])))
searchQueryIDFGrouped = searchQueryIDFMap.groupByKey()

# COMMAND ----------

searchQueryJoinTFIDF = searchQueryIDFNormalized.leftOuterJoin(TFIDF)
searchQueryJoinTFIDF.collect()

# COMMAND ----------

searchQueryJoinTFIDF = searchQueryJoinTFIDF.map(lambda x : (x[1][1], (x[0], x[1][0])))
searchQueryJoinTFIDF.collect()

# COMMAND ----------

searchQueryJoinTFIDFKeyAdjust = searchQueryJoinTFIDF.map(lambda x :((x[1][1]), (x[0], x[1][0])))
searchQueryJoinTFIDFKeyAdjust.collect()

# COMMAND ----------

searchQueryJoinTFIDFGrouped = searchQueryJoinTFIDFKeyAdjust.groupByKey() #.mapValues(list)
searchQueryJoinTFIDFGrouped.collect()

# COMMAND ----------

# Here we have vector for each document with matching terms and corresponding TFIDF values
# Now use this for cosine similarity comparison against original search query
searchQueryJoinTFIDFGrouped.map(lambda x: (x[0] ,[ (term[0], term[1]) for term in x[1]])).collect()

# COMMAND ----------

searchQueryJoinVector = searchQueryJoinTFIDFGrouped.map(lambda x: (x[0] ,[ (term[1]) for term in x[1]]))
searchQueryJoinVector.collect()

# COMMAND ----------

searchQueryIDFGrouped.map(lambda x: (x[0] ,[ (term[0], term[1]) for term in x[1]])).collect()

# COMMAND ----------

searchQueryIDValues = searchQueryIDFGrouped.map(lambda x : ([((term[0]), (term[1])) for term in x[1]]))
searchQueryIDValues.collect()

# COMMAND ----------

searchQueryVector = searchQueryIDFGrouped.map(lambda x : ([( (term[1])) for term in x[1]]))
searchQueryVector = searchQueryVector.flatMap(lambda x : (x))
searchQueryVector.collect()

# COMMAND ----------

# Cosine similarity calcultaion for (Query, Document[x]) = dot product (Query, Document[x]) / ||Query|| * |Docuement|
cartesianJoin =  searchQueryJoinVector.cartesian(searchQueryVector)
cartesianJoin.collect()

# COMMAND ----------


