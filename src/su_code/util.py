from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def create_test_submission(urlid, filename, prediction):
    content = ['urlid,label']
    for i, p in enumerate(prediction):
        content.append('%i,%f' % (urlid[i], p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'