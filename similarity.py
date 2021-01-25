class Tokenizer:
    def __init__(self, text, RemoveStopWords=True):
        self.text = text
        self.RemoveStopWords = RemoveStopWords
        self.punctuations = ['.', '!', '?', '\'', '\"',
                             ',', ':', ';', '(', ')', '[', ']', '<', '>', '\n']
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                           'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                           'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                           'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                           'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                           'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                           'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                           'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                           'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                           'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                           'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                           'now', 'm', 're', 'would', 'd', 'll']
        self.tokens = self.preprocess()

    def preprocess(self):
        text = self.text.lower()
        text = text.replace('-', '')
        for p in self.punctuations:
            text = text.replace(p, ' ')

        words = [w for w in text.split(' ') if w != '']
        self.total_words = len(words)
        self.stopWords = len([w for w in words if w in self.stop_words])

        if self.RemoveStopWords:
            tokens = [word for word in words if word not in self.stop_words]
        else:
            tokens = words

        return tokens


class Vectorizer:
    def __init__(self, RemoveStopWords=True):
        self.RemoveStopWords = RemoveStopWords
        self.vocabulary_ = {}
        self.dictionary_ = {}
        self.cv = []
        self.tfidf = []

    def fit(self, docs):
        self.preprocess_docs = [
            Tokenizer(doc, RemoveStopWords=self.RemoveStopWords).tokens for doc in docs]

        for doc in self.preprocess_docs:
            for word in doc:
                self.vocabulary_[word] = self.vocabulary_.get(word, 0) + 1

        self.dictionary_ = {w: i for i, w in enumerate(self.vocabulary_)}
        self.total_words = len(self.vocabulary_)
        self.total_docs = len(docs)
        return self

    def get_CountVectorizer(self):
        for doc in self.preprocess_docs:
            cv = [0]*self.total_words
            for w in doc:
                j = self.dictionary_[w]
                cv[j] += 1
            self.cv.append(cv)
        return self.cv

    def get_TFIDF(self):
        self.cv = self.get_CountVectorizer()
        for i in range(self.total_docs):
            tfidf = [0]*self.total_words
            for word in self.preprocess_docs[i]:
                j = self.dictionary_[word]
                tfidf[j] += self.cv[i][j]/self.vocabulary_[word]
            self.tfidf.append(tfidf)
        return self.tfidf


class TextSimilarity:
    def __init__(self, docs, RemoveStopWords=True, use_tfidf=False):
        self.docs = docs
        self.RemoveStopWords = RemoveStopWords
        self.use_tfidf = use_tfidf

    def get_cosin_similarity(self, v1, v2):
        # cos_sim = dot(a, b)/(norm(a)*norm(b))
        v1_dot_v2 = 0
        norm_v1 = 0
        norm_v2 = 0

        for i in range(len(v1)):
            v1_dot_v2 += v1[i]*v2[i]
            norm_v1 += v1[i]**2
            norm_v2 += v2[i]**2

        return v1_dot_v2/(norm_v1*norm_v2)**0.5

    def get_similarities(self):
        countVectorizer = Vectorizer(
            RemoveStopWords=self.RemoveStopWords).fit(self.docs)
        if self.use_tfidf:
            vec = countVectorizer.get_TFIDF()
        else:
            vec = countVectorizer.get_CountVectorizer()

        similarities = {}

        for i in range(countVectorizer.total_docs):
            for j in range(i+1, countVectorizer.total_docs):
                similarities[f'Text{i}-Text{j}'] = self.get_cosin_similarity(
                    vec[i], vec[j])
        return similarities
