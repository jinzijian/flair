import argparse
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--loss', default='THL', type=str, help="loss type, should be in [THL, CE，CTL]")

    parser.add_argument('--Oval', default='change', type=str, help="change O or not, should be in [change, not_change]")

    parser.add_argument('--weight_ture', default=0.5, type=float, help="weight of the positive threshold loss")

    parser.add_argument('--weight_combine', default=0.5, type=float, help="weight of the combine loss")

    parser.add_argument('--threshold', default= 0.01, type=float, help="bar of others and other classes")

    parser.add_argument('--margin', default=0.01, type=float, help="the margin of threshold")

    parser.add_argument('--lr', default=0.1, type=float, help="learning rate")

    parser.add_argument('--run', default='test', type=str, help="saving/logging directory")

    args = parser.parse_args()

    assert args.loss in ['THL', 'CE', 'THL-S']
    assert args.Oval in ['change', 'not_change']

    params_str = 'lr%fwg%fthl%fmg%f' % (args.lr,args.weight_ture,args.threshold,args.margin)


    # define columns
    columns = {0: 'text', 1: 'pos', 3: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = '/p300/flair/flair/mydata'



    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='eng.train.txt',
                                  test_file='eng.testb.txt',
                                  dev_file='eng.testa.txt')
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)

    # 4. initialize embeddings
    embedding_types = [

        WordEmbeddings('/p300/flair/flair/embeddings/glove.gensim'),

        # comment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use flair embeddings
        # FlairEmbeddings('news-forward'),
        # FlairEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=False,
                                            threshold=args.threshold,
                                            margin=args.margin,
                                            weight_true=args.weight_ture,
                                            loss_type= args.loss,
                                            Oval = args.Oval)


    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/{}'.format(args.run),
                  learning_rate=args.lr,
                  mini_batch_size=32,
                  max_epochs= 30,
                  params_str = params_str)



