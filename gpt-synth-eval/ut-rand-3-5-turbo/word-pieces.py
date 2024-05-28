import os
import subprocess
import sentencepiece as spm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", dest="train", default=False, help="Used if splitting is not needed")
    parser.add_argument("--split", action="store_true", dest="split", default=False, help="Used if training is not needed")
    parser.add_argument("--decode", action="store_true", dest="decode", default=False, help="Used if need splitting is not needed")
    parser.add_argument("--size", type=int, dest="size", default=32000, help="Number of wordpieces")
    parser.add_argument("--corpora", dest="corpora", nargs="+", help="File names of all preprocessed files for both languages separated by spaces")
    parser.add_argument("--model", dest="model", help="Wordpiece model file prefix or prefix of an existing model", default="wordpieces")

    args = parser.parse_args()

    if args.train:
        print("Creating a training file")
        subprocess.call("cat "+' '.join(args.corpora)+"| shuf | head -n 10000000 > "+args.model+".train", shell=True)
        print("Starting training")
        spm.SentencePieceTrainer.Train(input=args.corpora, model_prefix=args.model, vocab_size=args.size, input_sentence_size=10000000,  shuffle_input_sentence=True)
        subprocess.call("rm "+args.model+".train", shell=True)

    sp = spm.SentencePieceProcessor()

    print("Loading model")
    sp.Load(args.model+".model")

    if args.split:
        for corpus in args.corpora:
            print("splitting file", corpus)
            with open(corpus, 'r') as f:
                sentences = f.readlines()
            with open(os.path.join(os.path.split(corpus)[0], 'spm' + '-' + os.path.split(corpus)[1]), 'w') as f:
                for sentence in sentences:
                    f.write(' '.join([x for x in sp.EncodeAsPieces(sentence)]))
                    f.write('\n')

    if args.decode:
        for corpus in args.corpora:
            print("Secoding file", corpus)
            with open(corpus, 'r') as f:
                sentences = f.readlines()
            with open(os.path.join(os.path.split(corpus)[0], 'dspm' + '-' + os.path.split(corpus)[1]), 'w') as f:
                for sentence in sentences:
                    f.write(sp.DecodePieces(sentence.strip('\n').split(' ')))
                    f.write('\n')
