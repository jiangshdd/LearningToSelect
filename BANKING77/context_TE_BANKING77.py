
import argparse
import csv
import logging
import json
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
                              
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import pairwise
from sentence_transformers import SentenceTransformer

from transformers import RobertaTokenizer
from transformers.optimization import AdamW
from transformers import RobertaModel#RobertaForSequenceClassification

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'



class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)


    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        # print('\noutputs_single:', outputs_single[0][:,0,:])
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)
        # print('hidden_states_single:', 
        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single



class RobertaClassificationHead(nn.Module):

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        # print('\nfeatures:', x)
        # print('feature size:', x.size())
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def load_all_data(data_path):
    example_list = []
    with open(data_path, newline='') as f:
        for row in f:
            utterance = row.split('\t')[1].strip()
            label = row.split('\t')[0].strip()
            positive_example = {'utterance': utterance, 'class': label}
            example_list.append(positive_example)
    return example_list

def load_categories(json_fname):
    f = open(json_fname, 'r')
    cat_list = json.load(f)
    return cat_list


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, premise_class=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.premise_class = premise_class


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, premise_class_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.premise_class_id = premise_class_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def load_train(self, round_list, args, train_path):
        examples_list = []
        class_list_up_to_now = []
        round_indicator_up_to_now = []
        for round in round_list:
            '''first collect the class set in this round'''
            examples_this_round = []
            class_set_in_this_round = set()
            filename = train_path
            # filename = '../data/banking_data/banking77/split/'+round+'/train.txt'

            readfile = codecs.open(filename, 'r', 'utf-8')
            for row in readfile:
                parts = row.strip().split('\t')
                assert len(parts)==2
                class_name = parts[0].strip()
                class_set_in_this_round.add(class_name)
            readfile.close()
            class_list_up_to_now += list(class_set_in_this_round)
            round_indicator_up_to_now+=[round]*len(class_set_in_this_round)
            '''transform each example into entailment pair'''
            filename = train_path
            # filename = '../data/banking_data/banking77/split/'+round+'/train.txt'

            readfile = codecs.open(filename, 'r', 'utf-8')
            for row in readfile:
                parts = row.strip().split('\t')
                assert len(parts)==2
                class_name = parts[0].strip()
                class_str = ' '.join(class_name.split('_'))
                # print('class_str:', class_str)
                example_str = parts[1].strip()
                '''positive pair'''
                examples_this_round.append( InputExample(guid=round, text_a=example_str, text_b=class_str, label='entailment', premise_class=class_name))
                '''negative pairs'''
                negative_class_set = set(class_set_in_this_round)-set([class_name])
                for negative_class in negative_class_set:
                    class_str = ' '.join(negative_class.split('_'))
                    examples_this_round.append( InputExample(guid=round, text_a=example_str, text_b=class_str, label='non-entailment', premise_class=class_name))

            readfile.close()
            examples_list.append(examples_this_round)
        return examples_list, class_list_up_to_now, round_indicator_up_to_now


    def load_dev_or_test(self, round_list, seen_classes, flag, dev_path):
        examples_rounds = []
        example_size_list = []
        for round in round_list:
            examples = []
            instance_size = 0
            filename = dev_path
            # filename = '../data/banking_data/banking77/split/'+round+'/'+flag+'.txt'

            readfile = codecs.open(filename, 'r', 'utf-8')
            for row in readfile:
                parts = row.strip().split('\t')
                assert len(parts)==2
                class_name = parts[0].strip()
                if round == 'ood':
                    class_name = 'ood'
                example_str = parts[1].strip()

                for seen_class in seen_classes:
                    '''each example compares with all seen classes'''
                    class_str = ' '.join(seen_class.split('_'))
                    examples.append(
                        InputExample(guid=flag, text_a=example_str, text_b=class_str, label='entailment', premise_class=class_name))
                instance_size+=1
            readfile.close()
            examples_rounds+=examples
            example_size_list.append(instance_size)
        return examples_rounds#, example_size_list



    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def convert_examples_to_features(examples, label_list, eval_class_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}
    class_map = {label : i for i, label in enumerate(eval_class_list)}
    # print('label_map:', label_map)
    # print('class_map:', class_map)

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc='writing example')):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]


        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              premise_class_id = class_map[example.premise_class]))
    return features

def convert_examples_to_features_concatenate(examples, label_list, eval_class_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 top_k_candidates = []):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}
    class_map = {label : i for i, label in enumerate(eval_class_list)}

    features = []
    group_start_idlist = [77 * i for i in range(len(examples)//77)]
    for idx, group_id in enumerate(tqdm(group_start_idlist, desc='writing concat example')):
        sub_examples = examples[group_id:group_id+77]

        for (ex_index, example) in enumerate(sub_examples):
            # if ex_index % 10000 == 0:
            #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = tokenizer.tokenize(example.text_b)
            '''something added'''
            # other_3_examples_in_the_group = [ex_i for ex_i in sub_examples if ex_i.text_b != example.text_b]
            top_k_for_this_example = [ex_i for ex_i in top_k_candidates[idx] if ex_i != example.text_b]
            # for ex_i in sub_examples:
            #     if ex_i.text_b != example.text_b:
            tokens_b_concatenated = []
            # tokens_b_concatenated.append(tokens_b+[sep_token]+tokens_b+[sep_token]+tokens_b+[sep_token]+tokens_b)
            for ii in range(2):
                random.shuffle(top_k_for_this_example)
                tail_seq = []
                for ex_i in top_k_for_this_example:
                    tail_seq += [sep_token]+tokenizer.tokenize(ex_i)+[sep_token]
                tokens_b_concatenated.append(tokens_b+[sep_token]+tail_seq)
            for tokens_b in tokens_b_concatenated:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
                special_tokens_count = 7 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]
                segment_ids = [sequence_a_segment_id] * len(tokens)


                tokens += tokens_b 
                segment_ids += [sequence_b_segment_id] * (len(tokens_b))


                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                if output_mode == "classification":
                    label_id = label_map[example.label]
                elif output_mode == "regression":
                    label_id = float(example.label)
                else:
                    raise KeyError(output_mode)

                # if ex_index < 5:
                #     logger.info("*** Example ***")
                #     logger.info("guid: %s" % (example.guid))
                #     logger.info("tokens: %s" % " ".join(
                #             [str(x) for x in tokens]))
                #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                #     logger.info("label: %s (id = %d)" % (example.label, label_id))

                features.append(
                        InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id,
                                      premise_class_id = class_map[example.premise_class]))
    return features



def get_top_K_candidates(similarity_model, examples_dict, all_class_list,  device, args):

    K = 60

    def embedding_similarity(intent_embeddings, class_name_embeddings):
        similarity = pairwise.cosine_similarity(intent_embeddings, class_name_embeddings)
        return similarity
        
    def sentence_bert_similarity(similarity_model, utterance_list, class_name_list, device):
        similarity_model.to(device)
        with torch.no_grad():
            class_name_embeddings = similarity_model.encode(class_name_list)
            intent_embeddings = similarity_model.encode(utterance_list)
        similarity = embedding_similarity(torch.tensor(intent_embeddings), torch.tensor(class_name_embeddings))
        return similarity

    utterance_list = [i['utterance'] for i in examples_dict]
    class_name_list = [i for i in all_class_list]

    class_similarity = sentence_bert_similarity(similarity_model, utterance_list, class_name_list, device)
    top_K_indice = [np.argsort(i)[-K:] for i in class_similarity]
    top_K_candidates = [np.array(class_name_list)[i] for i in top_K_indice]

    return top_K_candidates

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def examples_to_features(source_examples, label_list, eval_class_list, args, tokenizer, batch_size, output_mode, dataloader_mode='sequential'):
    source_features = convert_examples_to_features_concatenate(
        source_examples, label_list, eval_class_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    dev_all_input_ids = torch.tensor([f.input_ids for f in source_features], dtype=torch.long)
    dev_all_input_mask = torch.tensor([f.input_mask for f in source_features], dtype=torch.long)
    dev_all_segment_ids = torch.tensor([f.segment_ids for f in source_features], dtype=torch.long)
    dev_all_label_ids = torch.tensor([f.label_id for f in source_features], dtype=torch.long)
    dev_all_premise_class_ids = torch.tensor([f.premise_class_id for f in source_features], dtype=torch.long)

    dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids, dev_all_premise_class_ids)
    if dataloader_mode=='sequential':
        dev_sampler = SequentialSampler(dev_data)
    else:
        dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    return dev_dataloader

def main(args_task_name, args_cache_dir, args_round_name, 
         args_max_seq_length, args_do_train, args_do_eval, 
         args_do_lower_case, args_train_batch_size, 
         args_eval_batch_size, args_learning_rate, 
         args_num_train_epochs, args_warmup_proportion, 
         args_no_cuda, args_local_rank, args_seed, 
         args_gradient_accumulation_steps, args_fp16, 
         args_loss_scale, args_server_ip, args_server_port, args_train_file):

    parser = argparse.ArgumentParser()
 
    ## Required parameters
    parser.add_argument("--task_name",
                        default=args_task_name,
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    ## Other parameters
    parser.add_argument("--train_file",
                        default=args_train_file,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--cache_dir",
                        default=args_cache_dir,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--round_name",
                        default=args_round_name,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--result_name",
                        type=str,
                        help="result output file name")  

    parser.add_argument("--max_seq_length",
                        default=args_max_seq_length,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        default=args_do_train,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        default = args_do_eval,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        default = args_do_lower_case,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=args_train_batch_size,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=args_eval_batch_size,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=args_learning_rate,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=args_num_train_epochs,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=args_warmup_proportion,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        default=args_no_cuda,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=args_local_rank,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=args_seed,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=args_gradient_accumulation_steps,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        default = args_fp16,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, 
                        default=args_loss_scale,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default=args_server_ip, help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default=args_server_port, help="Can be used for distant debugging.")
    # parser.add_argument('--time_stamp',
    #                     help="Time stamp to store results")

    parser.add_argument('-f')
 
    args = parser.parse_args()

    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    round_name_2_rounds={'r1':['n1']}

    
    train_path = root_dir + '/data/banking_data/banking77/split/n1/' +str(args.train_file) +'.txt'
    dev_path = root_dir + 'data/banking_data/banking77/split/n1/dev.txt'
    test_path = root_dir + 'data/banking_data/banking77/split/n1/test.txt'

    """ load data  """
    category_path = root_dir + 'data/banking_data/categories.json'
    all_class_list = load_categories(category_path)
    all_class_list = [' '.join(i.split('_')) for i in all_class_list]

    train_example_list = load_all_data(train_path)
    """ Top-K selection """
    """ load top-K selection model """
    similarity_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    train_top_K_candidates = get_top_K_candidates(similarity_model, train_example_list, all_class_list,  device, args)


    model = RobertaForSequenceClassification(3)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.load_state_dict(torch.load(mnli_model), strict=False)
    # model.load_state_dict(torch.load('../data/MNLI_pretrained.pt', map_location='cpu'), strict=False)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
  
    round_list = round_name_2_rounds.get(args.round_name)
    '''load training in list'''
    train_examples_list, train_class_list, train_class_2_split_list = processor.load_train(round_list, args, train_path) # no odd training examples
    # print('train_class_list:', train_class_list)
    assert len(train_class_list) == len(train_class_2_split_list)
    # assert len(train_class_list) ==  20+(len(round_list)-2)*10
    '''dev and test'''
    dev_examples = processor.load_dev_or_test(round_list, train_class_list, 'dev', dev_path)
    test_examples = processor.load_dev_or_test(round_list, train_class_list, 'test', test_path)

    # test_examples = processor.load_dev_or_test(round_list, train_class_list, 'test')
    # print('train size:', [len(train_i) for train_i in train_examples_list], ' dev size:', len(dev_examples), ' test size:', len(test_examples))
    entail_class_list = ['entailment', 'non-entailment']
    eval_class_list = train_class_list
    test_split_list = train_class_2_split_list
    train_dataloader_list = []
    for train_examples in train_examples_list:
        train_features = convert_examples_to_features(
            train_examples, entail_class_list, eval_class_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        train_features_concatenate = convert_examples_to_features_concatenate(
            train_examples, entail_class_list, eval_class_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0, top_k_candidates = train_top_K_candidates)#4 if args.model_type in ['xlnet'] else 0,)
        train_features+=train_features_concatenate

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_premise_class_ids = torch.tensor([f.premise_class_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_premise_class_ids)
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        train_dataloader_list.append(train_dataloader)


    # dev_dataloader = examples_to_features(dev_examples, entail_class_list, eval_class_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='sequential')
    # test_dataloader = examples_to_features(test_examples, entail_class_list, eval_class_list, args, tokenizer, args.eval_batch_size, "classification", dataloader_mode='sequential')
        
    '''load dev set'''
    dev_features = convert_examples_to_features(
        dev_examples, entail_class_list, eval_class_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    dev_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    dev_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    dev_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    dev_all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
    dev_all_premise_class_ids = torch.tensor([f.premise_class_id for f in dev_features], dtype=torch.long)


    dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids, dev_all_premise_class_ids)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)


    '''load test set'''
    test_features = convert_examples_to_features(
        test_examples, entail_class_list, eval_class_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_all_premise_class_ids = torch.tensor([f.premise_class_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids, test_all_premise_class_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    '''training'''


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)

    max_test_acc = 0.0
    max_dev_acc = 0.0
    for round_index, round in enumerate(round_list):
        '''for the new examples in each round, train multiple epochs'''
        train_dataloader = train_dataloader_list[round_index]
        for epoch_i in range(args.num_train_epochs):
            for _, batch in enumerate(tqdm(train_dataloader, desc="train|"+round+'|epoch_'+str(epoch_i))):
                model.train()
                batch = tuple(t.to(device) for t in batch)
    
                input_ids, input_mask, _, label_ids, premise_class_ids = batch

                logits = model(input_ids, input_mask)
      
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
                # print("\nloss:", loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            '''evaluation'''
            dev_acc= evaluate(model, device, round_list, test_split_list, train_class_list, args, dev_examples, dev_dataloader, 'dev')
            print(f'dev acc: {dev_acc}, max dev acc: {max_dev_acc}')
            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                test_acc = evaluate(model, device, round_list, test_split_list, train_class_list, args, test_examples, test_dataloader, 'test')
                print(f'test acc: {test_acc}')
                print(f'TRAIN FILE: {args.train_file}')

    print(f'!!!!!!!!!!final test acc: {test_acc} !!!!!!!!!!!!!!!!!\n')

def evaluate(model, device, round_list, test_split_list, train_class_list, args, dev_examples, dev_dataloader, flag):
    '''evaluation'''
    model.eval()

    logger.info(f"***** Running {flag} *****")
    logger.info("  Num examples = %d", len(dev_examples))

    preds = []
    gold_class_ids = []
    for _, batch in enumerate(tqdm(dev_dataloader, desc="test")):
        input_ids, input_mask, segment_ids, label_ids, premise_class_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        gold_class_ids+=list(premise_class_ids.detach().cpu().numpy())

        with torch.no_grad():
            logits = model(input_ids, input_mask)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = softmax(preds[0],axis=1)
    pred_label_3way = np.argmax(preds, axis=1) #dev_examples, 0 means "entailment"
    pred_probs = list(preds[:,0]) #prob for "entailment" class: (#input, #seen_classe)
    assert len(pred_label_3way) == len(dev_examples)
    assert len(pred_probs) == len(dev_examples)
    assert len(gold_class_ids) == len(dev_examples)

    pred_label_3way = np.array(pred_label_3way).reshape(len(dev_examples)//len(train_class_list),len(train_class_list))
    
    pred_probs = np.array(pred_probs).reshape(len(dev_examples)//len(train_class_list),len(train_class_list))
    gold_class_ids = np.array(gold_class_ids).reshape(len(dev_examples)//len(train_class_list),len(train_class_list))

    '''verify gold_class_ids per row'''
    rows, cols = gold_class_ids.shape
    for row in range(rows):
        assert len(set(gold_class_ids[row,:]))==1
    gold_label_ids = list(gold_class_ids[:,0])
    pred_label_ids_raw = list(np.argmax(pred_probs, axis=1))
    pred_max_prob = list(np.amax(pred_probs, axis=1))
    pred_label_ids = []
    for idd, seen_class_id in enumerate(pred_label_ids_raw):
        pred_label_ids.append(seen_class_id)

    assert len(pred_label_ids) == len(gold_label_ids)
    acc_each_round = []

    for round_name_id in round_list:
        #base, n1, n2, ood
        round_size = 0
        rount_hit = 0
        if round_name_id != 'ood':
            for ii, gold_label_id in enumerate(gold_label_ids):
                if test_split_list[gold_label_id] == round_name_id:
                    round_size+=1
                    if gold_label_id == pred_label_ids[ii]:
                        rount_hit+=1
            acc_i = rount_hit/round_size
            acc_each_round.append(acc_i)

    final_test_performance = acc_each_round[0]
    # print('\nfinal_test_performance:', final_test_performance)
    return final_test_performance

if __name__ == "__main__":
    args_task_name = 'rte'
    args_cache_dir = ''
    args_round_name = 'r1'
    args_max_seq_length = 450
    args_do_train = True
    args_do_eval = False
    args_do_lower_case = True
    args_train_batch_size = 8
    args_eval_batch_size = 64
    args_learning_rate = 1e-6
    args_num_train_epochs = 5
    args_warmup_proportion = 0.1
    args_no_cuda = False
    args_local_rank = -1
    args_seed = 42
    args_gradient_accumulation_steps = 1
    args_fp16 = False
    args_loss_scale = 0
    args_server_ip = ''
    args_server_port = ''
    args_train_file = 'one_shot_0'
    
    """ LOCAL  """
    root_dir = './'
    mnli_model =  '../model/MNLI_pretrained.pt'

    main(args_task_name, args_cache_dir, args_round_name, 
         args_max_seq_length, args_do_train, args_do_eval, 
         args_do_lower_case, args_train_batch_size, 
         args_eval_batch_size, args_learning_rate, 
         args_num_train_epochs, args_warmup_proportion, 
         args_no_cuda, args_local_rank, args_seed, 
         args_gradient_accumulation_steps, args_fp16, 
         args_loss_scale, args_server_ip, args_server_port, args_train_file)