from nltk.corpus.reader import PlaintextCorpusReader
source_dir = "Stock"
pcr = PlaintextCorpusReader(root=source_dir, fileids=".*\.txt")

from nltk.probability import FreqDist
fd = FreqDist(samples=pcr.words())
print(fd.most_common(n=100))



from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
corpus = PathLineSentences(source_dir)#"Gossiping")

model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
print(model.wv.most_similar(positive=['高端',"長榮"], negative=['疫苗']))
#iOS test data : print(model.wv.most_similar(positive=['手機',"iPhone"], negative=['平板'])) 


# 2. By using the GenSim Word2Vec module, 
#    find a word x in an analogy like "man : king :: woman : x" (read: man is to king as woman is to x) 
#    in your PTT texts.
#    (Please upload your Python script to GitHub, and paste your GitHub link on the online text of eCourse homework.)
