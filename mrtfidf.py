from __future__ import division
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep
from math import log
import re 

# match any alphanumeric character with a multiplity of {1..n}
# i.e. words, and the ' character
WORD_RE = re.compile("r[\w']+")
class MRTFIDF(MRJob):
    INPUT_PROTOCOL = JSONValueProtocol
    
    # output => (word, docName, D), 1
    def mapper_get_words(self, _, json_doc):
        D = len(json_doc)
        for i in range(D):
            docContent = json_doc[i]
            #for word in docContent['userContent'].split():
            for word in WORD_RE.findall(docContent['userContent']):
                yield (word.lower(),docContent['userId'],D), 1
    
    # compute term frequency for each document (number of times a word appear in a doc)
    # (word, docName, D), n
    def reducer_count_words_per_doc(self, docInfo,occurences):
            yield (docInfo[0],docInfo[1], docInfo[2]), sum(occurences)
    
    # => docname, (word,n,D)
    def mapper_total_number_of_words_per_docs(self, docInfo, n):
         yield docInfo[1], (docInfo[0],n, docInfo[2])
    
    # compute the total number of terms in each doc =>(word,docName, D), (n,N)
    def reducer_total_number_of_words_per_docs(self,docName,words_per_doc):
        total = 0
        n = []
        word = []
        D = []
        for value in words_per_doc:
            total += value[1]
            n.append(value[1])
            word.append(value[0])
            D.append(value[2])
            # n = value[1]
            # word = value[0]
        N = [total]*len(word)
        
        for value in range(len(word)):
            yield (word[value], docName, D[value]), (n[value], N[value])
            
    # word frequency in corpus
    # => (word, (docname, n, N, D, 1))
    def mapper_number_of_documents_a_word_appear_in(self, wordInfo, wordCounts):
        yield wordInfo[0], (wordInfo[1], wordCounts[0], wordCounts[1],wordInfo[2] ,1)
    
    # number of documents in the corpus in which the word appears
    # => ((word, docname, D), (n, N, m))
    def reducer_word_frequency_in_corpus(self, word, wordInfoAndCounts):
        total = 0
        docName = []
        n = []
        N = []
        D = []
        
        for value in wordInfoAndCounts:
            total += 1
            docName.append(value[0])
            n.append(value[1])
            N.append(value[2])
            D.append(value[3])
   
        # we need to compute the total numbers of documents in corpus
        m = [total]* len(n)
        
        for value in range(len(m)):
            yield (word, docName[value], D[value]), (n[value], N[value], m[value])
    
    # compute tf-idf 
    # ((word, docname, D), (n, N, m)) => ((word, docname), TF*IDF)
    def mapper_calculate_tf_idf(self, wordInfo, wordMetrics):
        tfidf = (wordMetrics[0] / wordMetrics[1]) * log(wordInfo[2] / wordMetrics[2])
        yield (wordInfo[0], wordInfo[1]), tfidf
        
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   reducer=self.reducer_count_words_per_doc),
            MRStep(mapper=self.mapper_total_number_of_words_per_docs,
                  reducer=self.reducer_total_number_of_words_per_docs),
            MRStep(mapper=self.mapper_number_of_documents_a_word_appear_in,
                  reducer=self.reducer_word_frequency_in_corpus),
            MRStep(mapper=self.mapper_calculate_tf_idf)
        ]

    
if __name__ == '__main__':
    MRTFIDF.run()
