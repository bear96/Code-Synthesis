# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import argparse
import bleu
import weighted_ngram_match
import syntax_match
import dataflow_match
import pandas as pd
import os

parser = argparse.ArgumentParser()
# parser.add_argument('--refs', type=str, nargs='+', required=True,
#                         help='reference files')
# parser.add_argument('--hyp', type=str, required=True, 
#                         help='hypothesis file')
# parser.add_argument('--lang', type=str, required=True, 
#                         choices=['java','js','c_sharp','php','go','python','ruby', 'sql'],
#                         help='programming language')
# parser.add_argument('--params', type=str, default='0.25,0.25,0.25,0.25',
#                         help='alpha, beta and gamma')
parser.add_argument('--model', type=str, default='plbart', choices=['plbart', 'codet5', 'codebert', 'codegpt', 'all'])
parser.add_argument('--language', type=str, default='both', choices=['python', 'sql', 'both'])
parser.add_argument('--ref_trunc', type=bool, default=False)
args = parser.parse_args();

def code_bleu(refs, hyp, lang, params='0.25,0.25,0.25,0.25'):

    alpha,beta,gamma,theta = [float(x) for x in params.split(',')]

    # preprocess inputs
    pre_references = [[x.strip() for x in open(refs, 'r', encoding='utf-8').readlines()]]
    hypothesis = [x.strip() for x in open(hyp, 'r', encoding='utf-8').readlines()]

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references)*len(hypothesis)


    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('keywords/'+lang+'.txt', 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, 'python')

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, 'python')

    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
                        format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score

    print('CodeBLEU score: ', code_bleu_score)
    return code_bleu_score

eval_file = "eval_scores.csv"
eval_models = ['plbart', 'codebert', 'codet5', 'codegpt']
if os.path.isfile(eval_file):
    eval_scores= pd.read_csv(eval_file)
else:
    eval_scores = pd.DataFrame([[0,0,0,0],[0,0,0,0]])
    eval_scores.columns = eval_models
    eval_scores.index = ['python', 'sql']
if args.model in eval_models:
    eval_models = [args.model]
for model in eval_models:
    if args.language == 'python' or args.language == 'both' :  
        print("Calculating {}'s python CodeBleu Score...'".format(model))
        ref_file = "./test_files/{}_python/reference.txt".format(model) if model == 'codebert' else\
        "./test_files/{}_Reference_PY.txt".format("Full" if not args.ref_trunc else "Truncated")
        eval_scores.loc['python', model] = code_bleu("./test_files/{}_python/hypothesis.txt".format(model),\
                                                 ref_file, "python")
    print("")    
    
    if args.language == 'sql' or args.language == 'both':
        print("Calculating {}'s sql CodeBleu Score...'".format(model))
        ref_file = "./test_files/{}_sql/reference.txt".format(model) if model == 'codebert' else\
        "./test_files/{}_Reference_SQL.txt".format("Full" if not args.ref_trunc else "Truncated")
        eval_scores.loc['sql', model] = code_bleu("./test_files/{}_sql/hypothesis.txt".format(model),\
                                               ref_file, "sql")

    print("")
eval_scores.to_csv(eval_file)

