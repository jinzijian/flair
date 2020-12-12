from flair.data import Sentence
from flair.embeddings import BertEmbeddings

# instantiate BERT embeddings
bert_embeddings = BertEmbeddings()

# make example sentence
sentence = Sentence('I love Berlin.', use_tokenizer=True)

# embed sentence
bert_embeddings.embed(sentence)

# print embedded tokens
for token in sentence:
    print(token)
    print(token.embedding)