import argparse
import csv
import logging
import json
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
                              
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from scipy.special import softmax
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import RobertaTokenizer
from transformers.optimization import AdamW
from transformers import RobertaModel#RobertaForSequenceClassification

from torch.utils.data import Dataset

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'


class TypingDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = []

        with open(label_file, "r", encoding="utf-8") as fin:
            label_lst = []
            for lines in fin:
                lines = lines.split()[0]
                lines = ' '.join(lines.split('_'))
                label_lst.append(lines)
            self.label_lst = label_lst
            self.general_lst = label_lst[0:9]
            self.fine_lst = label_lst[9:130]
            self.ultrafine_lst = label_lst[130:]

        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        for line in lines:
            line = json.loads(line)

            premise = line['premise']
            entity = line['entity']
            # could truncate generated annotation
            annotation = line['annotation']
            annotation = [' '.join(a.split('_')) for a in annotation]

            top_k = line['top_k']
            top_k = [' '.join(a.split('_')) for a in top_k]

            pos_in_top_k = list(set(annotation).intersection(set(top_k)))

            annotation_general = list(set(annotation).intersection(set(self.general_lst)))
            annotation_fine = list(set(annotation).intersection(set(self.fine_lst)))
            annotation_ultrafine = list(set(annotation).intersection(set(self.ultrafine_lst)))

            self.data.append([premise, entity, annotation, annotation_general, annotation_fine, annotation_ultrafine, pos_in_top_k, top_k])


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


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

def load_categories(label_file):
    with open(label_file, "r", encoding="utf-8") as fin:
        label_lst = []
        for lines in fin:
            lines = lines.split()[0]
            lines = ' '.join(lines.split('_'))
            label_lst.append(lines)
    return label_lst



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, entity = None, label=None):
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
        self.text_a = text_a
        self.text_b = text_b
        self.entity = entity
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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
    def load_train(self, data_file, label_file):
        examples = []

        with open(label_file, "r", encoding="utf-8") as fin:
            label_lst = []
            for lines in fin:
                lines = lines.split()[0]
                lines = ' '.join(lines.split('_'))
                label_lst.append(lines)
            self.label_lst = label_lst
            self.general_lst = label_lst[0:9]
            self.fine_lst = label_lst[9:130]
            self.ultrafine_lst = label_lst[130:]

        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        top_k_candidates = []
        group_start_idlist = [0]
        for _, line in enumerate(tqdm(lines, desc='constructing training pairs')):
            line = json.loads(line)

            premise = line['premise']
            entity = line['entity']
            # could truncate generated annotation
            annotation = line['annotation']
            annotation = [' '.join(a.split('_')) for a in annotation]
            top_k = line['top_k']
            top_k = [' '.join(a.split('_')) for a in top_k]

            top_k_candidates.append(top_k)

            pos_in_top_k = list(set(annotation).intersection(set(top_k)))

            annotation_general = list(set(annotation).intersection(set(self.general_lst)))
            annotation_fine = list(set(annotation).intersection(set(self.fine_lst)))
            annotation_ultrafine = list(set(annotation).intersection(set(self.ultrafine_lst)))

            for typing in annotation:
                examples.append( InputExample(text_a=premise, text_b=typing, entity = entity, label='entailment'))
            
            # negative_class_set = set(self.label_lst)-set(annotation)
            # negative_class_set = [tmp for tmp in self.label_lst if tmp not in annotation]
            negative_class_set = set(top_k)-set(annotation)

            for typing in negative_class_set:
                examples.append( InputExample(text_a=premise, text_b=typing, entity = entity, label='non-entailment'))
            
            next_start = group_start_idlist[-1] + len(annotation) + len(negative_class_set)
            group_start_idlist.append(next_start)

        return examples, top_k_candidates, group_start_idlist


    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]




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

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc='writing example')):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        hypo = ' '.join([example.entity, 'is a', example.text_b +'.'])

        tokens_b = tokenizer.tokenize(hypo)
     
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
                              label_id=label_id))

    
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
                                 top_k_candidates = [], group_start_idlist = []):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}
    max_len = 0

    features = []
    # group_start_idlist = [75 * i for i in range(len(examples)//75)]
    for idx, group_id in enumerate(tqdm(zip(group_start_idlist, group_start_idlist[1:]), desc='writing concat example', total=len(group_start_idlist[1:]))):
        # print(group_id)
        sub_examples = examples[group_id[0]:group_id[1]]

        for (ex_index, example) in enumerate(sub_examples):
            # if ex_index % 10000 == 0:
            #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.text_a)

            hypo = ' '.join([example.entity, 'is a', example.text_b +'.'])

            tokens_b = tokenizer.tokenize(hypo)
            '''something added'''
            # other_3_examples_in_the_group = [ex_i for ex_i in sub_examples if ex_i.text_b != example.text_b]
            top_k_for_this_example = [ex_i for ex_i in top_k_candidates[idx] if ex_i != example.text_b]
            # for ex_i in sub_examples:
            #     if ex_i.text_b != example.text_b:
            tokens_b_concatenated = []
            # tokens_b_concatenated.append(tokens_b+[sep_token]+tokens_b+[sep_token]+tokens_b+[sep_token]+tokens_b)
            for ii in range(1):
                prob = random.random()
                if prob <= 0.6:
                    random.shuffle(top_k_for_this_example)
                    tail_seq = []
                    for ex_i in top_k_for_this_example:
                        tail_seq += [sep_token]+tokenizer.tokenize(ex_i)+[sep_token]
                    tokens_b_concatenated.append(tokens_b+[sep_token]+tail_seq)

            for tokens_b in tokens_b_concatenated:
            
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
                max_len = max(max_len, len(input_ids)) 

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
                                      label_id=label_id))
    print(f'!!!!!!!max len: {max_len}!!!!!!!!')
    return features


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
    parser.add_argument("--threshold",
                        default='0,1',
                        type=str,
                        help="Threshold range.")
    parser.add_argument('--server_ip', type=str, default=args_server_ip, help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default=args_server_port, help="Can be used for distant debugging.")

    parser.add_argument('-f')
 
    args = parser.parse_args()
    args.threshold = [float(i) for i in args.threshold.split(',')] 

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

    train_path = './data/' +str(args.train_file) +'.json'
    dev_path ='./data/dev_processed.json'
    test_path = './data/test_processed.json'

    """ load data  """
    category_path = './data/types.txt'

    model = RobertaForSequenceClassification(3)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.load_state_dict(torch.load(mnli_model), strict=False)

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
  
    '''load training in list'''
    train_examples_list, train_top_K_candidates, group_start_idlist = processor.load_train(train_path, category_path) # no odd training examples
    '''dev and test'''
    # dev data
    dev_dataset = TypingDataset(dev_path, category_path)
    # test data
    test_dataset = TypingDataset(test_path, category_path)

    entail_class_list = ['entailment', 'non-entailment']
    eval_class_list = []

    train_features = convert_examples_to_features(
        train_examples_list, entail_class_list, eval_class_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)
    print(f'num train features: {len(train_features)}')

    train_features_concatenate = convert_examples_to_features_concatenate(
        train_examples_list, entail_class_list, eval_class_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0, top_k_candidates = train_top_K_candidates, group_start_idlist = group_start_idlist)
    print(f'num train features concat: {len(train_features_concatenate)}')

    train_features+=train_features_concatenate

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    '''training'''

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)

    max_test_acc = 0.0
    max_dev_acc = 0.0

    for epoch_i in range(args.num_train_epochs):
        for _, batch in enumerate(tqdm(train_dataloader, desc='train|epoch_'+str(epoch_i))):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, _, label_ids = batch

            logits = model(input_ids, input_mask)
    
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
            # print("\nloss:", loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        '''evaluation'''
        evaluate(args, dev_dataset, model, tokenizer, epoch_i, device)
 
def evaluate(args, eval_dataset, model, tokenizer, global_step, device):
    model.eval()

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: zip(*x))

    dev_output_scores = []
    dev_all_top_k = []
    dev_all_truth = []

    dev_epoch_iterator = tqdm(eval_dataloader, desc="evaluating")
    for step, batch in enumerate(dev_epoch_iterator):
        premise_lst, entity_lst, pos_lst, pos_general_lst, pos_fine_lst, pos_ultrafine_lst, pos_in_top_k, top_k = [list(item) for item in batch]
        data_combo = []

        for idx in range(len(premise_lst)):
            premise = premise_lst[idx]
            entity = entity_lst[idx]
            truth_typing = pos_lst[idx]
            top_k_typing = top_k[idx]
            top_k_typing = [' '.join(a.split('_')) for a in top_k_typing]

            for typing in top_k_typing:

                input_temp = ' '.join([premise, 2*tokenizer.sep_token, entity, 'is a', typing+'.'])
                data_combo.append(input_temp)
            
        # true
        model_inputs = tokenizer(data_combo, padding=True, return_tensors='pt')
        model_inputs = model_inputs.to(device)

        input_ids = model_inputs['input_ids']
        input_mask = model_inputs['attention_mask']
        with torch.no_grad():
            output = model(input_ids, input_mask)
            output = nn.functional.softmax(output, dim=-1)[:, 0] 
        dev_output = output.reshape(len(premise_lst), len(top_k[0]))
        dev_output_scores.append(dev_output)
        dev_all_top_k += top_k
        dev_all_truth  += pos_lst

        # for candidate, i in zip(top_k, pred_indicator):
        #     preds.append([candidate[i]])
    dev_output_scores = torch.cat(dev_output_scores, dim=0)

    dev_threshold_list = np.arange(args.threshold[0], args.threshold[1], 0.002)
    dev_macro_performance_list =  metric(dev_threshold_list, dev_all_top_k, dev_output_scores, dev_all_truth, len(dev_output_scores))
    dev_performance = dev_macro_performance_list[0]
    
    print('dev_performance:', dev_performance)


def macro(gold, pred, len_test):
    """ adopt from UFET paper codes """
    def f1(p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)
    num_examples = len_test
    p = 0.
    r = 0.
    pred_example_count = 0.
    pred_label_count = 0.
    gold_label_count = 0.

    for true_labels, predicted_labels in zip(gold, pred):
        if predicted_labels:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels):
            gold_label_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    if pred_example_count > 0:
        precision = p / pred_example_count
    if gold_label_count > 0:
        recall = r / gold_label_count
    avg_elem_per_pred = pred_label_count / pred_example_count
    return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)

def metric(threshold_list, all_top_k, output_scores, all_truth, len_set):
    macro_performance_list = []
    for _, th in enumerate(tqdm(threshold_list, desc = 'searching threshold')):
        try:
            pred_indicator = torch.where(output_scores>th, True, False).cpu().numpy()

            preds = []
            for candidate, i in zip(all_top_k, pred_indicator):
                preds.append(list(np.array(candidate)[i]))

            count, pred_count, avg_pred_count, macro_precision, macro_recall, macro_f1 = macro(all_truth, preds, len_set)

            macro_performance = (('threshold', f"{th:.4}"), ('P', f"{macro_precision:.2%}"), ('R', f"{macro_recall:.2%}"), ('F1', f"{macro_f1:.2%}"))
            # print(macro_performance)
            
            macro_performance_list.append(macro_performance)
        except:
            pass
   
    macro_performance_list.sort(key = lambda x: -float(x[3][1].replace('%', 'e-2')))
    return macro_performance_list


if __name__ == "__main__":
    args_task_name = 'rte'
    args_cache_dir = ''
    args_round_name = 'r1'
    args_max_seq_length = 410
    args_do_train = True
    args_do_eval = False
    args_do_lower_case = True
    args_train_batch_size = 8
    args_eval_batch_size = 16
    args_learning_rate = 5e-6
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
    args_train_file = 'train_processed'

    """ LOCAL  """
    mnli_model =  './MNLI_pretrained.pt'

    main(args_task_name, args_cache_dir, args_round_name, 
         args_max_seq_length, args_do_train, args_do_eval, 
         args_do_lower_case, args_train_batch_size, 
         args_eval_batch_size, args_learning_rate, 
         args_num_train_epochs, args_warmup_proportion, 
         args_no_cuda, args_local_rank, args_seed, 
         args_gradient_accumulation_steps, args_fp16, 
         args_loss_scale, args_server_ip, args_server_port, args_train_file)