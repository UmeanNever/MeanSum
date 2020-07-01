from gensim.corpora import Dictionary


class ExtraWordFilter(object):
    def __init__(self):
        self.dct = None
        self.stopwords = None

    def fit(self, docs, no_above, **kwargs):
        segmented_docs = [doc.lower().split() for item in docs for doc in item]
        self.dct = Dictionary(segmented_docs)
        self.dct.filter_extremes(no_above=no_above, **kwargs)
        print("Extra Dct size:{}".format(len(self.dct.token2id)))
        # print("Dct keys: {}".format(self.dct.token2id.keys()))
        return self.dct.token2id

    def transform(self, docs):
        segmented_docs = [[doc.split() for doc in item] for item in docs]
        transformed_docs = [[" ".join([word for word in doc if word.lower() in self.dct.token2id.keys()]) for doc in
                             item] for item in segmented_docs]
        return transformed_docs

    def fit_transform(self, docs, no_above, **kwargs):
        self.fit(docs, no_above, **kwargs)
        return self.transform(docs)
