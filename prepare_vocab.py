"""
Prepare vocabulary and initial word vectors.
"""
import json
import msgpack
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import vocab, constant, helper


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('data_dir', help='TACRED directory.')
    parser.add_argument('squad_dir', help='SQuAD directory.')
    parser.add_argument('vocab_dir', help='Output vocab directory.')
    parser.add_argument('--glove_dir', default='dataset/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')

    args = parser.parse_args()
    return args


def process_squad(squad_msgpack):
    train, dev = squad_msgpack
    train_tokens = []
    dev_tokens = []
    for row in train:
        train_tokens += row[1]  # context
        train_tokens += row[5]  # question
    for row in dev:
        dev_tokens += row[1]
        dev_tokens += row[5]
    return train_tokens, dev_tokens


def main():
    args = parse_args()

    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'
    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)
    # processing squad intermediate files
    with open(args.squad_dir + '/intermediate.msgpack', 'rb') as squad_file:
        squad_msgpack = msgpack.load(squad_file, encoding='utf-8')
    squad_train, squad_dev = squad_msgpack
    squad_train_tokens, squad_dev_tokens = process_squad(squad_msgpack)
    if args.lower:
        train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in \
                                                 (train_tokens, dev_tokens, test_tokens)]

        squad_train_tokens, squad_dev_tokens = [[t.lower() for t in tokens] for tokens in \
                                                (squad_train_tokens, squad_dev_tokens)]
    # load glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))

    print("building vocab...")
    v = build_vocab(train_tokens + squad_train_tokens, glove_vocab, args.min_freq)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov * 100.0 / total))

    print("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    print("all done.")

    print('converting SQuAD dataset to ids')

    id2word = v
    word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])

    def to_id(row, unk_id=constant.UNK_ID):
        context_tokens = row[1]
        context_features = row[2]
        context_tags = row[3]
        context_ents = row[4]
        question_tokens = row[5]
        question_ids = [word2id[w] if w in word2id else unk_id for w in question_tokens]
        context_ids = [word2id[w] if w in word2id else unk_id for w in context_tokens]
        tag_ids = [constant.POS_TO_ID[w] if w in constant.POS_TO_ID else unk_id for w in context_tags]
        ent_ids = [constant.NER_TO_ID[w] if w in constant.NER_TO_ID else unk_id for w in context_ents]
        return [row[0], context_ids, context_features, tag_ids, ent_ids, question_ids] + row[6:]

    squad_train = list(map(to_id, squad_train))
    squad_dev = list(map(to_id, squad_dev))
    result = {
        'train': squad_train,
        'dev': squad_dev
    }
    # train: id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer_start, answer_end
    # dev:   id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer
    with open('dataset/SQuAD/data.msgpack', 'wb') as f:
        msgpack.dump(result, f)


def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            tokens += d['tokens']
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens


def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + entity_masks() + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v


def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total - matched


def entity_masks():
    """ Get all entity mask tokens as a list. """
    masks = []
    subj_entities = list(constant.SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())[2:]
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks


if __name__ == '__main__':
    main()
